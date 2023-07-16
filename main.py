# This file references the pytorch lightning template, with the following license:
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from datetime import datetime
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import dataset
import models
import utils
import wandb

logger = utils.get_logger()


class Model(pl.LightningModule):
    def __init__(self, args, trainer):
        super().__init__()
        if wandb.run is not None:
            wandb.run.log_code(".")

        preferred_linalg_library = getattr(args, "preferred_linalg_library", None)
        if preferred_linalg_library is not None and preferred_linalg_library != "do_not_set":
            torch.backends.cuda.preferred_linalg_library(args.preferred_linalg_library)
            logger.info(f"Setting preferred linalg library to {args.preferred_linalg_library}")

        if args.rank <= 0:
            os.makedirs(args.checkpoints_dir,
                        exist_ok=args.allow_overwriting_checkpoints_dir)
            if not args.test:
                os.makedirs(os.path.join(args.checkpoints_dir, "saved"),
                            exist_ok=args.allow_overwriting_checkpoints_dir)
            # We may have validation in training
            os.makedirs(os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_dir_name", "saved_eval")),
                        exist_ok=args.allow_overwriting_checkpoints_dir)
            os.makedirs(os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_export_dir_name", "saved_eval_export")),
                        exist_ok=args.allow_overwriting_checkpoints_dir)
        if args.rank >= 0:
            torch.cuda.set_device(args.rank)
            trainer.strategy.barrier()

            self.world_size = trainer.strategy.world_size
        else:
            self.world_size = 1

        dataset_cls_name = getattr(args, "dataset_cls", "VideoDataset")
        self.dataset_cls = dataset.__dict__[dataset_cls_name]
        logger.info(f"Using dataset: {dataset_cls_name}")

        self.save_hyperparameters(args, ignore="model")
        self.args = args
        self.model = models.__dict__[args.model_cls](args, **args.model_kwargs)

        if args.pretrained_model is not None:
            if "*" in args.pretrained_model:
                potential_matches = glob(args.pretrained_model)
                assert len(
                    potential_matches) == 1, f"{potential_matches} is not unique"
                args.pretrained_model = potential_matches[0]
            logger.info(
                f"Loading pretrained model from {args.pretrained_model}")
            ckpt = torch.load(args.pretrained_model, map_location="cpu")

            # If the checkpoint is the main model, we load the main model.
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            if getattr(self.args, "pretrained_model_backbone_only", False):
                state_dict = {k:v for k, v in state_dict.items() if "backbone" in k}
            example_key = list(state_dict.keys())[0]
            
            ema_loaded = False
            ema_in_model = hasattr(self.model, 'backbone2_ema') and self.model.backbone2_ema is not None
            if example_key.startswith("model."):
                # Main model
                # ema exists in the state_dict
                ema_in_state_dict = len([k for k in state_dict.keys() if '_ema' in k])
                if ema_in_model and not ema_in_state_dict:
                    assert self.model.decode_head2_ema is not None, "decode_head2_ema is not enabled with backbone2_ema enabled"
                    logger.info("Detected EMA in model but not in state_dict, loading state_dict to both the main model and ema")
                    ema_state_dict = {
                        k.replace("backbone2", "backbone2_ema").replace("decode_head2", "decode_head2_ema"): v for k, v in state_dict.items() if "backbone2" in k or "decode_head2" in k
                    }
                    state_dict_with_ema = {**state_dict, **ema_state_dict}
                else:
                    # Do not need to modify the state_dict
                    state_dict_with_ema = state_dict
                
                if getattr(self.args, "drop_head_decode_head2", False):
                    logger.info("Dropping decode_head2 (with ema)")
                    state_dict_with_ema = {k: v for k, v in state_dict_with_ema.items() if "decode_head2" not in k}
                
                mismatches = self.load_state_dict(state_dict_with_ema, strict=False)
                ema_loaded = True
            elif example_key.startswith("module."):
                # Moco model
                model_prefix = 'module.encoder_q'
                
                for k in list(state_dict.keys()):
                    # retain only student model up to before the embedding layer
                    if k.startswith(model_prefix) and not k.startswith(model_prefix + '.fc'):
                        # remove prefix
                        new_key = k.replace(model_prefix + '.', "")
                        state_dict[new_key] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                mismatches = self.model.backbone2.load_state_dict(
                    state_dict, strict=False)
            elif 'conv1.weight' in state_dict.keys():
                # DenseCL model
                mismatches = self.model.backbone2.load_state_dict(state_dict, strict=False)
            elif 'backbone2.conv1.weight' in state_dict.keys():
                # Only the model in the main model
                mismatches = self.model.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError("Unknown module in state_dict")

            if not ema_loaded:
                # To implement, call `self.model.init_ema()`
                assert (getattr(self.model, "backbone2_ema", None) is None) and (getattr(self.model, "decode_head2_ema", None) is None), "EMA is enabled but weights are not loaded"

            logger.info(f"Mismatches in the pretrained model: {mismatches}")
        else:
            logger.info("Not loading pretrained model")

        self.accumulate_training_loss = {}

        self.object_channel = args.object_channel if args.object_channel is not None else os.environ.get("OBJECT_CHANNEL", None)
        if isinstance(self.object_channel, str):
            self.object_channel = int(self.object_channel)
        # self.args.object_channel is for global use
        self.args.object_channel = self.object_channel
        logger.info(f"Using {self.object_channel} as object channel")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        losses = self.forward(x)
        if isinstance(losses, dict):
            loss = losses['loss']
        else:
            loss = losses
            losses = dict(loss=loss)
        
        for loss_key, loss_value in losses.items():
            if 'loss' not in loss_key:
                continue
            self.accumulate_training_loss[loss_key] = self.accumulate_training_loss.get(loss_key, 0.) + loss_value.item()
            if (batch_idx + 1) % self.args.loss_log_interval == 0:
                self.log(f"train_{loss_key}", self.accumulate_training_loss[loss_key] /
                        self.args.loss_log_interval, sync_dist=True, reduce_fx="mean")
                self.accumulate_training_loss[loss_key] = 0.

        if torch.isnan(loss):
            raise Exception("loss is NaN")
        return loss

    @rank_zero_only
    def on_validation_start(self):
        self.on_test_start()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        # We use max IoU channel in validation to be fast.
        # In test, we should only use one channel.
        self.test_step(batch, batch_idx)

    # Need to run it on non-zero rank to make sure we have val_miou key
    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs, name="val_miou", display_all=False)

    @rank_zero_only
    def on_test_start(self):
        self.iou_all_sequences = {}
        self.max_channel_freq = [0 for _ in range(self.args.model_kwargs["mask_layer"])]

    @rank_zero_only
    def test_step(self, batch, batch_idx, always_use_max_iou_channel=False):
        x = batch
        pred_masks_current_batch = self.forward(x)
        assert len(pred_masks_current_batch) == len(
            batch['ann']), f"{len(pred_masks_current_batch)} != {len(batch['ann'])}"
        pred_masks_current_batch_resize = utils.eval_utils._resize(
            pred_masks_current_batch, batch['ann'].shape[1:3])
        # -1 means use hard max (turn masks into one hot)
        num_channels = pred_masks_current_batch_resize.shape[1]
        if self.args.eval_pos_th != -1:
            pred_masks_current_batch_resize = (
                pred_masks_current_batch_resize > self.args.eval_pos_th).long().cpu().numpy()
        else:
            pred_masks_current_batch_resize_max_idx = pred_masks_current_batch_resize.argmax(
                dim=1)
            pred_masks_current_batch_resize = F.one_hot(
                pred_masks_current_batch_resize_max_idx,
                num_classes=num_channels
            ).permute((0, 3, 1, 2)).long().cpu().numpy()

        ignore_locations = batch['ann'] == 128
        anns_np = (batch['ann'] / 255).long()
        anns_np[ignore_locations] = -1
        anns_np = anns_np.cpu().numpy()
        for pred_mask_resize, ann, seq_name in zip(pred_masks_current_batch_resize, anns_np, batch['seq_names']):
            # pred_mask_resize and ann are numpy array
            # index 1 selects the iou in the foreground
            if always_use_max_iou_channel or (self.object_channel is None):
                frame_ious = [
                    utils.iou(pred_mask_resize_item, ann, num_classes=2, ignore_index=-1)[1] for pred_mask_resize_item in pred_mask_resize]
                max_channel = np.argmax(frame_ious)
                self.max_channel_freq[max_channel] += 1
                frame_iou = frame_ious[max_channel]
            else:
                frame_iou = utils.iou(
                    pred_mask_resize[self.object_channel], ann, num_classes=2, ignore_index=-1)[1]
            iou_current_sequence = self.iou_all_sequences.setdefault(
                seq_name, [])
            iou_current_sequence.append(frame_iou)

    def test_epoch_end(self, outputs, name="test_miou", display_all=True):
        if (self.object_channel is None) and (not self.trainer.sanity_checking) and ((self.current_epoch >= getattr(self.args, "set_object_channel_after_epoch", 1) - 1) or self.trainer.testing):
            # Set object channel only once (if `always_use_max_iou_channel` is set, object channel will be ignored, otherwise this will always be used)
            if self.args.rank >= 0: # Distributed
                if self.args.rank == 0:
                    object_channel_current_rank = np.argmax(self.max_channel_freq)
                else:
                    object_channel_current_rank = 0

                object_channel = torch.tensor([object_channel_current_rank], device="cuda")
                dist.all_reduce(object_channel, op=dist.ReduceOp.SUM)
                object_channel = object_channel.item()
            else:
                object_channel = np.argmax(self.max_channel_freq)

            self.object_channel = object_channel
            self.args.object_channel = self.object_channel
            if self.args.rank <= 0:
                print(f"Rank {self.args.rank}: Set object channel to {self.object_channel} (Max channel distribution at local rank: {self.max_channel_freq})")
            else:
                print(f"Rank {self.args.rank}: Set object channel to {self.object_channel}")
        
        if self.args.rank > 0:
            # Otherwise model checkpoint will not work in validation
            # Sync values to make model checkpoint correctly.
            self.log(name, 0., sync_dist=True, reduce_fx="sum")
            self.log(name + "_frame_avg", 0., sync_dist=True, reduce_fx="sum")
            return

        miou_each_sequence = {}
        iou_sum = 0.
        iou_num_frames = 0.
        for seq_name, miou_current_sequence in self.iou_all_sequences.items():
            miou = np.nanmean(miou_current_sequence).astype(np.float32)
            miou_each_sequence[seq_name] = miou

            iou_sum += np.sum(miou_current_sequence).astype(np.float32)
            iou_num_frames += len(miou_current_sequence)

            if display_all:
                logger.info(f"{name}_{seq_name}: {miou * 100.:.2f}")
            # This is only computed and logged on rank 0, so do not sync.
            self.log(f'{name}_{seq_name}', miou, sync_dist=False)

        # We should not get NaN here unless some videos are empty or have all NaNs
        mean_miou_all_sequences = np.mean(list(miou_each_sequence.values())).astype(np.float32)

        logger.info(f"{name}: {mean_miou_all_sequences * 100.:.2f}")
        self.log(name, mean_miou_all_sequences, sync_dist=True, reduce_fx="sum")

        miou_frame_avg = iou_sum / iou_num_frames
        logger.info(f"{name}_frame_avg: {miou_frame_avg * 100.:.2f}")
        self.log(name + "_frame_avg", miou_frame_avg, sync_dist=True, reduce_fx="sum")

    def get_lr(self, epoch, power, base_lr, min_lr):
        coeff = (1 - epoch / self.args.epochs) ** power
        lr = (base_lr - min_lr) * coeff + min_lr
        return lr / base_lr

    def configure_optimizers(self):
        params = list(self.model.parameters())
        params_require_grad = [param for param in params if param.requires_grad]
        logger.info(f"Number of param tensors: {len(params)}. Number of param tensors (require grad): {len(params_require_grad)}.")
        optimizer = torch.optim.__dict__[self.args.optimizer](
            params_require_grad,
            lr=self.args.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        def lr_lambda(epoch): return self.get_lr(
            epoch, base_lr=self.args.learning_rate, **self.args.lr_scheduler_kwargs)
        return [optimizer], [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)]

    def on_train_epoch_start(self):
        logger.info(
            f"LR: {self.optimizers(False).state_dict()['param_groups'][0]['lr']}")

    def train_dataloader(self):
        train_dataset = self.dataset_cls(
            self.args.data_path, training=True,
            transform=dataset.get_transform(self.args, training=True),
            **self.args.dataset_kwargs, **self.args.train_dataset_kwargs)
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True) if self.args.multi_gpu else None
        
        if getattr(self.args, 'force_no_shuffle', False):
            assert not self.args.multi_gpu, "force_no_shuffle is for visualization with one GPU"
            shuffle = False
        else:
            shuffle = sampler is None
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle
        )

    def val_dataloader(self, subsample_frame_interval=10):
        # subsample_frames for fast evaluation
        test_data_path = self.args.test_data_path if getattr(self.args, 'test_data_path', None) else self.args.data_path
        val_dataset = self.dataset_cls(
            test_data_path, training=False,
            transform=dataset.get_transform(self.args, training=False),
            subsample_frame_interval=subsample_frame_interval,
            **self.args.dataset_kwargs, **self.args.test_dataset_kwargs)
        
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(val_dataset),
            shuffle=False
        )

    def test_dataloader(self):
        test_data_path = self.args.test_data_path if getattr(self.args, 'test_data_path', None) else self.args.data_path
        test_dataset = self.dataset_cls(
            test_data_path, training=False,
            transform=dataset.get_transform(self.args, training=False),
            **self.args.dataset_kwargs, **self.args.test_dataset_kwargs)
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.args.workers,
            pin_memory=True,
            sampler=torch.utils.data.SequentialSampler(test_dataset),
            shuffle=False
        )


exp_name = None


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items["v_num"] = exp_name
        return items


def main():
    global logger, exp_name

    parser = argparse.ArgumentParser(description='Train segmentation.')
    parser.add_argument('config', metavar='C', type=str, nargs='?',
                        help='path to config', default='configs/rcf/rcf_stage1.yaml')
    parser.add_argument('--test', help='test only',
                        default=False, action="store_true")
    parser.add_argument('--test-override-pretrained', help='override pretrained model and checkpoints directory at test',
                        default=None, type=str)
    parser.add_argument('--test-override-object-channel', help='override object channel at test',
                        default=None, type=int)
    parser.add_argument('--no-test', help='no test at the end of training',
                        default=False, action="store_true")
    # From detectron2
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    cli_args = parser.parse_args()
    config_path = cli_args.config
    test = cli_args.test
    no_test = cli_args.no_test

    rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if rank <= 0:
        utils.set_loglevel(debug=True)
    else:
        utils.set_loglevel(debug=False)

    logger.info(f"Loading config from {config_path}")

    args = utils.load_args(config_path, cli_opts=cli_args.opts)
    logger.info(str(args))

    # Use config file name as the experiment name
    # exp_name = config_path.split(
    #     "/")[-1][:-5] + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Use checkpoints_dir as the experiment name because we may override checkpoints_dir with CLI options
    exp_name = args.checkpoints_dir.split(
        "/")[-1] + "_" + datetime.now().strftime("%y%m%d_%H%M%S")
    wandb_logger = pl.loggers.WandbLogger(
        project="RCF", mode="disabled" if args.disable_wandb else None, name=exp_name, settings=wandb.Settings(start_method="thread"))
    # save_on_train_epoch_end should be set to True if there is no validation set
    # even on rank non-zero we need to have the monitor key (saving is ignored in Stratrgy class so other code that requires the monitor key will run on rank non-zero)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir, save_on_train_epoch_end=False, every_n_epochs=1, monitor='val_miou_frame_avg',
        save_top_k=2, save_last=True, mode='max', auto_insert_metric_name=True)

    trainer_cfg = {
        "logger": wandb_logger,
        # override_max_epochs is for early stopping without influencing the learning rate scheduler
        "max_epochs": getattr(args, "override_max_epochs", args.epochs),
        "accelerator": "gpu",
        "replace_sampler_ddp": False,
        "callbacks": [CustomProgressBar(), checkpoint_callback],
        **args.trainer_kwargs
    }

    args.config_path = config_path
    args.test = test
    args.rank = rank
    args.multi_gpu = rank > -1

    if args.multi_gpu:
        assert not test, "Testing with multi-GPUs is unsupported."
        trainer_cfg.update(dict(strategy="ddp_find_unused_parameters_false"))
    else:
        trainer_cfg.update(dict(devices=1))

    if test:
        if cli_args.test_override_pretrained is not None:
            args.pretrained_model = cli_args.test_override_pretrained
            args.checkpoints_dir = os.path.dirname(args.pretrained_model)
            logger.info(f"Overriding pretrained_model to {args.pretrained_model}")
        if cli_args.test_override_object_channel is not None:
            args.object_channel = cli_args.test_override_object_channel
            logger.info(f"Overriding object channel to {args.object_channel}")

    trainer = pl.Trainer(**trainer_cfg, default_root_dir=args.checkpoints_dir)
    model = Model(args, trainer)

    logger.info(f"{model}")

    if not test:
        trainer.fit(model=model)
        if not no_test:
            # Use hard max to test at the end
            args.saved_eval_dir_name = 'saved_eval_test'
            args.eval_pos_th = -1
            trainer.test(model=model)
    else:
        trainer.test(model=model)


if __name__ == "__main__":
    main()
