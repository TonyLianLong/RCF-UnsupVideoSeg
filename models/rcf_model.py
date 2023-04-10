import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torchvision
import flow_vis
import numpy as np
import utils

from mmseg.ops import resize

from pytorch_lightning.utilities import rank_zero_only

from .resnet import ResNet
from .flow_aggregation_head_with_residual import FlowAggregationHeadWithResidual
from .fcn_head import FCNHead
from .compactness_head import CompactnessHead
from .crf_head import CRFHead

from copy import deepcopy

logger = utils.get_logger()

class RCFModel(nn.Module):
    """RCFModel
    """
    def __init__(self,
                 args,
                 backbone2,
                 decode_head,
                 decode_head2,
                 decode_head3,
                 compactness_head=None,
                 crf_head=None,
                 crf_use_ema=False,
                 ema_m=0.999,
                 w_seg=2.0,
                 w_sharpen=0,
                 t_sharpen=0.25,
                 w_entropy=0,
                 w_compactness=0,
                 w_pl=0,
                 pl_pos_weight=1.,
                 pl_neg_weight=1.,
                 pl_mask_pos_th=0.35,
                 w_crf=0,
                 crf_pos_weight=1.,
                 crf_neg_weight=1.,
                 crf_mask_pos_th=-1.,
                 mask_layer=1,
                 train_iter=0,
                 train_cfg=None,
                 test_cfg=None,
                 align_corners=False,
                 mask_size=(48, 48),
                 log_interval=50,
                 freeze_backbone=False,
                 object_aware_sharpening=False,
                 separate_residual=False,
                 allow_mask_resize=False):
        super(RCFModel, self).__init__()
        self.args = args
        self.save_dir = os.path.join(args.checkpoints_dir, "saved")
        self.save_dir_eval = os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_dir_name", "saved_eval"))
        self.save_dir_eval_export = os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_export_dir_name", "saved_eval_export"))

        self.backbone2, self.backbone2_ema = self.create_backbone_with_ema(backbone2)
        
        # Moved from decoder in v1 to main module
        self.align_corners = align_corners

        self.mask_layer = mask_layer
        # decode_head is flow aggregation module
        self.decode_head = globals()[decode_head.pop('type')](args=self.args, **decode_head)
        # decode_head2 is for getting the masks
        self.decode_head2, self.decode_head2_ema = self.create_decode_head_with_ema(decode_head2)
        self.num_classes = self.decode_head2.num_classes
        # decode_head3 is for getting the residual
        self.decode_head3 = self.create_decode_head(decode_head3)

        self.w_compactness = w_compactness
        if compactness_head:
            self.compactness_head = globals()[compactness_head.pop('type')](args=self.args, **compactness_head)
            assert self.w_compactness != 0, "Compactness head is used but weight is 0"
        else:
            self.compactness_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

        if freeze_backbone:
            for _, param in self.backbone2.named_parameters():
                param.requires_grad_(False)

        self.train_iter = train_iter
        self.train_iter_per_log = 50

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.L1loss = nn.MSELoss()
        self.L1 = nn.L1Loss()

        self.w_seg = w_seg
        self.w_sharpen = w_sharpen
        self.t_sharpen = t_sharpen
        self.w_entropy = w_entropy
        
        assert not (w_sharpen != 0 and w_entropy != 0), "Only one of w_entropy and w_sharpen could be nonzero"
        
        self.w_pl = w_pl
        if self.w_pl > 0:
            assert self.args.object_channel is not None, "Pseudo label loss requires an object channel to be set at the beginning"
            self.pl_pos_weight = pl_pos_weight
            self.pl_neg_weight = pl_neg_weight
            self.pl_mask_pos_th = pl_mask_pos_th
        
        self.w_crf = w_crf
        if crf_head:
            self.crf_head = globals()[crf_head.pop('type')](args=self.args, **crf_head)
            assert self.w_crf != 0, "CRF head is used but weight is 0"
            
            # CRF is binary, but it is non-binary after resize.
            self.crf_pos_weight = crf_pos_weight
            self.crf_neg_weight = crf_neg_weight
            # Use non-binary by default for compatibility (crf_mask_pos_th=-1)
            self.crf_mask_pos_th = crf_mask_pos_th
        else:
            self.crf_head = None
        
        self.crf_use_ema = crf_use_ema
        
        self.ema_m = ema_m
        
        self.log_interval = log_interval

        self.mask_size = tuple(mask_size)
        self.allow_mask_resize = allow_mask_resize

        self.object_aware_sharpening = object_aware_sharpening
        
        # separate_residual: use different output conv for fw and bw residual
        self.separate_residual = separate_residual
        
        self.eval_on_ema = getattr(self.args, 'eval_on_ema', False)
        
        if self.eval_on_ema:
            assert self.backbone2_ema is not None and self.decode_head2_ema is not None, "Eval on EMA requires EMA to be enabled"
            logger.info("Evaluating EMA model")
        
        self.init_ema()
        
    def init_ema(self):
        if self.backbone2_ema is not None:
            utils.copy_param_and_buffer(src=self.backbone2, dest=self.backbone2_ema)
        if self.decode_head2_ema is not None:
            utils.copy_param_and_buffer(src=self.decode_head2, dest=self.decode_head2_ema)

    def create_backbone_with_ema(self, backbone2_kwargs: dict):
        # Has side effect: will remove "type" and "create_ema" key in backbone2
        backbone2_type = backbone2_kwargs.pop('type')
        backbone2_create_ema = backbone2_kwargs.pop('create_ema', False)
        backbone2 = globals()[backbone2_type](**backbone2_kwargs)
        if backbone2_create_ema:
            logger.info("Creating backbone2 with EMA")
            backbone2_ema = deepcopy(backbone2)
            utils.set_no_grad(backbone2_ema)
            backbone2_ema.cuda()
            backbone2_ema.eval()
        else:
            backbone2_ema = None

        return backbone2, backbone2_ema

    def create_decode_head_with_ema(self, decode_head_kwargs: dict):
        # Has side effect: will remove "type" and "create_ema" key in decode_head
        decode_head_type = decode_head_kwargs.pop('type')
        decode_head_create_ema = decode_head_kwargs.pop('create_ema', False)
        decode_head = globals()[decode_head_type](**decode_head_kwargs)
        if decode_head_create_ema:
            logger.info("Creating a decode head with EMA")
            decode_head_ema = deepcopy(decode_head)
            utils.set_no_grad(decode_head_ema)
            decode_head_ema.cuda()
            decode_head_ema.eval()
        else:
            decode_head_ema = None
        
        return decode_head, decode_head_ema

    def create_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = globals()[
            decode_head.pop('type')](**decode_head)
        return decode_head

    def init_weights(self):
        self.backbone2.init_weights()

    def _decode_head_forward(self, x, decode_head):
        """Run forward function and calculate loss for decode head in
        training."""
        pred = decode_head.forward(x)
        return pred

    def numpy_to_tensor(self, flowrgb):
        output = torch.from_numpy(flowrgb).cuda()
        output = torch.unsqueeze(output, 0).transpose(3, 2).transpose(2, 1)
        return output

    def resize(self, in_img, shape):
        output = resize(
            in_img,
            size=shape,
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        return output

    def let_tensor_vis(self, t, setmax=False):
        t = t.clone()
        if setmax:
            _th_ = 0.2
            t[:, 0, 0, 0] = _th_
            t[:, 1, 0, 0] = _th_
            assert torch.max(t[:, 0, :, :]) == _th_
            assert torch.max(t[:, 1, :, :]) == _th_
        t = torch.squeeze(t)
        t = t.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
        flow_color = flow_vis.flow_to_color(t, convert_to_bgr=False)
        flow_color = self.numpy_to_tensor(flow_color)
        return flow_color

    def extract_feat(self, imgs, net):
        """Extract features from images."""
        x = net(imgs)
        return x

    @rank_zero_only
    def save_eval_visualizations(self, tosave, paths, seq_ids, seq_names, name='eval', train_iter=0):
        try:
            # Filename uses idx_in_batch
            idx_in_batch = 0
            # 0 below means current_frame
            img_frame_id = paths[0][idx_in_batch].split('/')[-1][:-4]
            fn_name = f'{self.save_dir_eval}/{name}_{seq_names[idx_in_batch]}_{seq_ids[idx_in_batch]}_{img_frame_id}_{train_iter:07}.jpg'
            torchvision.utils.save_image(tosave, fn_name)
        except Exception as e:
            logger.warn(f"Error in saving: {fn_name} {e}")

    @rank_zero_only
    def export_seg(self, tosave, paths, seq_ids, seq_names, name='eval', train_iter=0, subdir=''):
        # tosave: [B, 3, mask_H, mask_W]
        # 0 means current frame
        if subdir:
            subdir += '/'
            if not os.path.exists(f'{self.save_dir_eval_export}/{subdir}'):
                os.makedirs(f'{self.save_dir_eval_export}/{subdir}')
        for idx_in_batch, (path, seq_name, seq_id) in enumerate(zip(paths[0], seq_names, seq_ids)):
            img_frame_id = path.split('/')[-1][:-4]
            fn_name = f'{self.save_dir_eval_export}/{subdir}{name}_{seq_name}_{img_frame_id}_{train_iter:07}.png'
            try:
                torchvision.utils.save_image(tosave[idx_in_batch], fn_name)
            except Exception as e:
                logger.warn(f"Error in saving: {fn_name} {e}")

    @rank_zero_only
    def export_all_seg(self, tosave, paths, seq_ids, seq_names, name='eval', train_iter=0):
        # A list with C items (C=5)
        for idx, tosave_item in enumerate(tosave):
            self.export_seg(tosave_item, paths, seq_ids, seq_names, name, train_iter, subdir=str(idx))

    def forward_eval(self, imgs, seq_ids, seq_names, paths, return_pred_vis_list=False):
        losses = dict()
        # Typically: _h: 392, _w: 697 (average between 384 and 400)
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        if self.eval_on_ema:
            all_feat = self.extract_feat(img_3, self.backbone2_ema)
            all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2_ema) # [B, 256, 98, 175], [B, 2048, 49, 88], [B, 512, 49, 88], [B, 1024, 49, 88]
        else:
            all_feat = self.extract_feat(img_3, self.backbone2)
            all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2) # [B, 256, 98, 175], [B, 2048, 49, 88], [B, 512, 49, 88], [B, 1024, 49, 88]
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(
            batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        ith_img = imgs[:, 0, :, :, :]
        ith_img = ith_img.detach()
        toH, toW = all_pred_mask.shape[-2:]
        ith_img_resize = self.resize(ith_img, (toH * 2, toW * 2))
        ith_img_resize = (ith_img_resize + 2.0) / 4.0

        # Export all masks
        pred_masks = all_pred_mask[:, 0, :, :, :]
        pred_vis_list = []

        for _i_ in range(min(self.mask_layer, pred_masks.shape[1])):
            _pred_mask_resize = self.resize(
                pred_masks[:, _i_:_i_+1, :, :], (toH * 2, toW * 2)).repeat(1, 3, 1, 1)
            pred_vis_list.append(_pred_mask_resize)
        tosave = torch.cat([ith_img_resize] + pred_vis_list, 2)

        if self.args.eval_save:
            self.save_eval_visualizations(tosave, paths, seq_ids, seq_names, train_iter=self.train_iter)
            if self.args.eval_export:
                # Note that saved images are resized to 2x (for visualization and here we did not change the resize)
                if getattr(self.args, "export_all_seg", False):
                    # Export all channels
                    self.export_all_seg(pred_vis_list, paths, seq_ids, seq_names, name='pred_seg', train_iter=self.train_iter)
                else: # Export object channel only
                    self.export_seg(pred_vis_list[self.args.object_channel], paths, seq_ids, seq_names, name='pred_seg', train_iter=self.train_iter)

        if return_pred_vis_list:
            return pred_masks, pred_vis_list

        return pred_masks

    def pred_separate_residual(self, feats, batch_size, im_num):
        # separate residual has different input compared to joint residual to make passing features with multiple resolutions easier
        # feats (before process): a list of [B * I, C, 48, 48]
        feats = [feat.unflatten(0, (batch_size, im_num)).flatten(1, 2) for feat in feats]
        # feats (processed): a list of [B, I * C, 48, 48]
        
        # all_pred_residual: [B, 2*2*C, 48, 48]
        # 2*2*C: [fw, bw] * [x, y] * C
        all_pred_residual = self._decode_head_forward(feats, self.decode_head3)
        # all_pred_residual_fw: [B, 2*C, 48, 48]
        # all_pred_residual_bw: [B, 2*C, 48, 48]
        all_pred_residual_fw, all_pred_residual_bw = all_pred_residual[:, :2*self.num_classes], all_pred_residual[:, 2*self.num_classes:]
        
        return all_pred_residual_fw, all_pred_residual_bw

    def pred_joint_residual(self, unflattened_feat):
        # feat: [B, I, 2048, 48, 48]

        # all_pred_residual_fw, all_pred_residual_bw: [B, 2*C, 48, 48]
        all_pred_residual_fw = self._decode_head_forward([unflattened_feat.flatten(1, 2)], self.decode_head3)
        all_pred_residual_bw = self._decode_head_forward([unflattened_feat[:, torch.tensor([1, 0], device=unflattened_feat.device)].flatten(1, 2)], self.decode_head3)

        # Alternative:
        # feat_img1, feat_img2 = feat[:, 0], feat[:, 1]
        # all_pred_residual_bw = self._decode_head_forward([torch.cat((feat_img2, feat_img1), dim=1)], self.decode_head3)

        return all_pred_residual_fw, all_pred_residual_bw

    def get_sharpen_loss(self, all_pred_mask, log_all_pred_mask, object_channel=None):
        # all_pred_mask: [B, I, C=5, 48, 48]
        if self.object_aware_sharpening:
            object_mask = all_pred_mask[:, :, object_channel]
            if False:
                # This is adding up the prob of non-object masks.
                non_object_mask = 1 - object_mask
                # The log is numerically unstable.
                # object_aware_mask: [B, I, C=2, 48, 48]
                object_aware_mask = torch.stack((object_mask, non_object_mask), dim=2)
                target = utils.sharpen(object_aware_mask.detach(), self.t_sharpen, dim=2)
                sharpen_loss = F.kl_div(object_aware_mask.log(), target, reduction="none")
                sharpen_loss = sharpen_loss.mean()
            else:
                # This is using hinge loss on the abs diff.
                all_pred_mask_except_object = all_pred_mask.detach().clone()
                all_pred_mask_except_object[:, :, object_channel] = 0.
                diff = (object_mask - all_pred_mask_except_object.max(dim=2).values).abs()
                sharpen_loss = (self.t_sharpen - diff).clamp(min=0)
                sharpen_loss = sharpen_loss.mean()
        else:
            target = utils.sharpen(all_pred_mask.detach(), self.t_sharpen, dim=2)
            sharpen_loss = F.kl_div(log_all_pred_mask, target, reduction="none")
            sharpen_loss = sharpen_loss.mean()
        return sharpen_loss

    def get_entropy_loss(self, all_pred_mask, log_all_pred_mask):
        # all_pred_mask: [B, I, C=5, 48, 48]
        return -(all_pred_mask * log_all_pred_mask).sum(dim=2).mean()

    def get_pl_loss(self, all_pred_mask, pl_masks):
        # all_pred_mask: [B, I, C=5, 48, 48]
        # pl_masks (range from 0 to 1): [B, I, 48, 48]
        if self.pl_mask_pos_th != -1:
            # pl_masks will be binary
            pl_masks = (pl_masks > self.pl_mask_pos_th).float()
        # object_channel_pred_mask: [B, I, 48, 48]
        object_channel_pred_mask = all_pred_mask[:, :, self.args.object_channel]

        # MSE with positive and negative weights
        pl_loss_pos = (torch.clamp(pl_masks - object_channel_pred_mask, min=0)) ** 2
        pl_loss_neg = (torch.clamp(pl_masks - object_channel_pred_mask, max=0)) ** 2
        pl_loss = pl_loss_pos.mean() * self.pl_pos_weight + pl_loss_neg.mean() * self.pl_neg_weight
        return pl_loss

    def get_crf_loss(self, all_pred_mask, crf_masks):
        # all_pred_mask: [B, I, C=5, 48, 48]
        # crf_masks (range from 0 to 1, binary): [B, I, 48, 48]
        if self.crf_mask_pos_th != -1.:
            # crf_masks will be binary
            crf_masks = (crf_masks > self.crf_mask_pos_th).float()
        # object_channel_pred_mask: [B, I, 48, 48]
        object_channel_pred_mask = all_pred_mask[:, :, self.args.object_channel]

        # MSE with positive and negative weights
        crf_loss_pos = (torch.clamp(crf_masks - object_channel_pred_mask, min=0)) ** 2
        crf_loss_neg = (torch.clamp(crf_masks - object_channel_pred_mask, max=0)) ** 2
        crf_loss = crf_loss_pos.mean() * self.crf_pos_weight + crf_loss_neg.mean() * self.crf_neg_weight
        return crf_loss

    def forward_train(self, imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks):
        # Usually, _h: 384, _w: 384 (random crop) (same for the flows)
        batch_size, im_num, num_channels, _h, _w = imgs.shape
        # Usually `I - 1` (I is im_num)
        flow_num = gt_fw_flows.shape[1]
        
        img_3 = imgs.view(batch_size * im_num, num_channels, _h, _w)
        # [B * I, 256, 96, 96], [B * I, 512, 48, 48], [B * I, 1024, 48, 48], [B * I, 2048, 48, 48]
        all_feat = self.extract_feat(img_3, self.backbone2)
        # [B * I, C=5, 48, 48]
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        if self.allow_mask_resize and (all_pred_mask.shape[-2:] != self.mask_size):
            all_pred_mask = self.resize(all_pred_mask, self.mask_size)
        # Change view in the feature: pass in the last feature map only
        if self.separate_residual:
            all_pred_residual_fw, all_pred_residual_bw = self.pred_separate_residual(all_feat, batch_size, im_num)
        else:
            # Only supports one resolution for joint residual
            all_pred_residual_fw, all_pred_residual_bw = self.pred_joint_residual(all_feat[-1].unflatten(0, (batch_size, im_num)))
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(
            batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
        # Take softmax across channel dimension (C=5 sum up to 1)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)
        log_all_pred_mask = F.log_softmax(all_pred_mask, dim=2)

        # mask shape should be (48, 48), all_pred_mask.shape[-3:]: (2, 48, 48)
        # gt_fw_flows, gt_bw_flows (before resize): [B, 1, C=2, 480, 584]
        gt_fw_flows = self.resize(gt_fw_flows.view(batch_size * flow_num, *gt_fw_flows.shape[2:]), self.mask_size)
        gt_bw_flows = self.resize(gt_bw_flows.view(batch_size * flow_num, *gt_bw_flows.shape[2:]), self.mask_size)
        # After resize and view: [B, 1, C=2, 48, 48]
        gt_fw_flows = gt_fw_flows.view(batch_size, flow_num, 2, *self.mask_size)
        gt_bw_flows = gt_bw_flows.view(batch_size, flow_num, 2, *self.mask_size)

        # Get flow from the flow head (pred_flows is normalized for visualization)
        pred_flows, loss_flow = self.decode_head(imgs, all_pred_mask, gt_fw_flows, gt_bw_flows, all_pred_residual_fw, all_pred_residual_bw)
        assert len(pred_flows['gt_flow']) == 1 and len(pred_flows['pred_flow']) == 1
        pred_flows_resize = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['pred_flow']][0]
        agg_flow_resize = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['agg_flow']][0]
        residual_adj_resize = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['residual_adj']][0]
        gt_flows_resize = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['gt_flow']][0]
        if len(pred_flows['affine_flow']) > 0:
            affine_flow_resize = [self.resize(it, all_pred_mask.shape[-2:]) for it in pred_flows['affine_flow']][0]
        else:
            affine_flow_resize = None

        if self.train_iter % self.log_interval == 0:
            mask_mat = [[all_pred_mask[:, it, layer_idx:layer_idx+1, :, :].repeat(
                1, 3, 1, 1) for it in range(im_num)] for layer_idx in range(self.mask_layer)]
            pic_mask_mat = [torch.cat(it, 0) for it in mask_mat]

            ith_imgs = []
            ith_flowxy_ori = []

        loss_input_warp_seg = loss_flow['seg']

        losses = {
            "loss_warp_seg": loss_input_warp_seg,
        }

        loss = loss_input_warp_seg * self.w_seg
        if self.w_sharpen > 0 and ((self.args.object_channel is not None) or (not self.object_aware_sharpening)):
            if self.object_aware_sharpening:
                loss_sharpen = self.get_sharpen_loss(all_pred_mask, log_all_pred_mask, object_channel=self.args.object_channel)
            else:
                loss_sharpen = self.get_sharpen_loss(all_pred_mask, log_all_pred_mask)
            loss = loss + loss_sharpen * self.w_sharpen
            losses["loss_sharpen"] = loss_sharpen
        elif self.w_entropy > 0:
            loss_entropy = self.get_entropy_loss(all_pred_mask, log_all_pred_mask)
            loss = loss + loss_entropy * self.w_entropy
            losses["loss_entropy"] = loss_entropy
        
        if self.compactness_head:
            loss_compactness = self.compactness_head.get_compactness_loss(all_pred_mask)
            if loss_compactness is not None:
                # It's None if we use object channel as compact channel (compact channel is set to -1), but we don't have object channel yet
                losses["loss_compactness"] = loss_compactness
                loss = loss + loss_compactness * self.w_compactness
        
        if self.w_pl > 0:
            pl_masks = self.resize(pl_masks, self.mask_size)
            loss_pl = self.get_pl_loss(all_pred_mask, pl_masks)
            losses["loss_pl"] = loss_pl
            loss = loss + loss_pl * self.w_pl
        
        if self.w_crf > 0:
            if self.crf_use_ema:
                # Note that the ema modules are run in eval time and could differ from the main module even if the input and the parameters are the same. Furthermore, there is dropout in decode_head2.
                # Switching to eval does not make them the same because the main module has an additional forward pass.
                # All the parameters should have requires_grad to be False, and the output of the EMA model should not have grad_fn.
                # [B * I, 256, 96, 96], [B * I, 512, 48, 48], [B * I, 1024, 48, 48], [B * I, 2048, 48, 48]
                all_feat_ema = self.extract_feat(img_3, self.backbone2_ema)
                # [B * I, C=5, 48, 48]
                all_pred_mask_ema = self._decode_head_forward(all_feat_ema, self.decode_head2_ema)
                all_pred_mask_ema = all_pred_mask_ema.view(
                    batch_size, im_num, self.mask_layer, _feat_h, _feat_w)
                # Take softmax across channel dimension (C=5 sum up to 1)
                all_pred_mask_ema = F.softmax(all_pred_mask_ema, dim=2)
                all_pred_mask_crf = all_pred_mask_ema
            else:
                all_pred_mask_crf = all_pred_mask
            # img_3: [B * I, 3, H, W]
            # Before resize: [B * I, 1, 96, 96]
            resized_mask = self.resize(all_pred_mask_crf.detach().flatten(0, 1)[:, self.args.object_channel:self.args.object_channel+1, ...], img_3.shape[-2:])
            # resized_mask[:, 0, ...]: [B * I, H, W]
            crf_masks = self.crf_head(img_3, resized_mask[:, 0, ...]).unflatten(0, (batch_size, im_num))
            # crf_masks (before): [B, I, H, W]
            # Resize CRF masks back to mask size:
            # crf_masks: [B, I, 96, 96]
            crf_masks = self.resize(crf_masks, self.mask_size)
            loss_crf = self.get_crf_loss(all_pred_mask, crf_masks)
            losses["loss_crf"] = loss_crf
            loss = loss + loss_crf * self.w_crf
        
        if self.backbone2_ema:
            utils.momentum_update_param_and_buffer(src=self.backbone2, dest=self.backbone2_ema, m=self.ema_m)
        
        if self.decode_head2_ema:
            utils.momentum_update_param_and_buffer(src=self.decode_head2, dest=self.decode_head2_ema, m=self.ema_m)
        
        losses["loss"] = loss

        line_mean_loss = 0.0
        for _i in range(im_num):
            ith_img = imgs[:, _i, :, :, :]
            ith_img = ith_img.detach()
            pred_mask = all_pred_mask[:, _i, :, :, :]

            if _i == 0:
                # The leading dimension is the batch dimension
                pred_flow = pred_flows_resize[:, :2, :, :]
                gt_flow = gt_flows_resize[:, :2, :, :]
                agg_flow_item = agg_flow_resize[:, :2, :, :]
                residual_adj_item = residual_adj_resize[:, :2, :, :]
                if affine_flow_resize is not None:
                    affine_flow_item = affine_flow_resize[:, :2, :, :]
                else:
                    affine_flow_item = None
            elif _i == 1:
                # The leading dimension is the batch dimension
                pred_flow = pred_flows_resize[:, 2:, :, :]
                gt_flow = gt_flows_resize[:, 2:, :, :]
                agg_flow_item = agg_flow_resize[:, 2:, :, :]
                residual_adj_item = residual_adj_resize[:, 2:, :, :]
                if affine_flow_item is not None:
                    affine_flow_item = affine_flow_resize[:, 2:, :, :]
                else:
                    affine_flow_item = None
            else:
                raise NotImplementedError("Only support im_num=2")

            if self.train_iter % self.log_interval == 0:
                for _batch_idx in range(batch_size):
                    flow_color_3_pred = self.let_tensor_vis(pred_flow[_batch_idx:_batch_idx+1])
                    flow_color_3_gt = self.let_tensor_vis(gt_flow[_batch_idx:_batch_idx+1])
                    flow_color_3_agg = self.let_tensor_vis(agg_flow_item[_batch_idx:_batch_idx+1])
                    flow_color_3_adj = self.let_tensor_vis(residual_adj_item[_batch_idx:_batch_idx+1])
                    
                    if affine_flow_item is not None:
                        flow_color_3_affine = self.let_tensor_vis(affine_flow_item[_batch_idx:_batch_idx+1])
                        # Order: Pred, GT, Agg, Affine, Adj
                        flow_color_3 = torch.cat([flow_color_3_pred, flow_color_3_gt, flow_color_3_agg, flow_color_3_affine, flow_color_3_adj], dim=2)
                    else:
                        flow_color_3 = torch.cat([flow_color_3_pred, flow_color_3_gt, flow_color_3_agg, flow_color_3_adj], dim=2)
                    ith_flowxy_ori.append(flow_color_3 / 255.0)
                ith_img_resize = self.resize(ith_img, pred_mask.shape[-2:])
                # This is an approximate un-normalization for visualization
                ith_imgs.append((ith_img_resize + 2.0)/4.0)

        if self.train_iter % self.log_interval == 0:
            # Filename uses idx_in_batch
            idx_in_batch = 0
            # 0 below means current_frame
            img_frame_id = paths[0][idx_in_batch].split('/')[-1][:-4]
            pic_ith_imgs = torch.cat(ith_imgs, 0)
            pic_ith_flowxy_ori = torch.cat(ith_flowxy_ori, 0)

            if pl_masks is not None:
                pl_masks_vis = [pl_masks[:, it:it+1, :, :].repeat(
                    1, 3, 1, 1) for it in range(im_num)]
                pl_masks_vis = torch.cat(pl_masks_vis, 0)
                
                tosave = torch.cat(pic_mask_mat + [pic_ith_imgs, pic_ith_flowxy_ori, pl_masks_vis], dim=2)
            else:
                tosave = torch.cat(pic_mask_mat + [pic_ith_imgs, pic_ith_flowxy_ori], dim=2)
            
            # Visualization:
            # Enable saving when log interval is 1
            if self.log_interval == 1:
                if self.train_iter == 0:
                    os.makedirs(f"{self.save_dir}_train_export", exist_ok=False)
                torch.save([imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks, tosave], f"{self.save_dir}_train_export/{seq_names[idx_in_batch]}_{img_frame_id}.pth")
            try:
                fn_name = '{}/train_iter{:07d}_{}_{}_{}_img_pred_recons.jpg'.format(
                    self.save_dir, self.train_iter, seq_names[idx_in_batch], seq_ids[idx_in_batch], img_frame_id)
                torchvision.utils.save_image(tosave, fn_name)
            except:
                logger.warn(f"Error in saving: {fn_name}")
        self.train_iter += 1

        return losses

    def forward(self, x, return_pred_vis_list=False):
        imgs, seq_ids, seq_names, paths = x['imgs'], x['seq_ids'], x['seq_names'], x['paths']
        # Stack images only (not annotations since they are not paired)
        imgs = torch.stack(imgs, dim=1)
        if self.training:
            gt_fw_flows, gt_bw_flows = torch.stack(x['gt_fw_flows'], dim=1), torch.stack(x['gt_bw_flows'], dim=1)
            if self.w_pl > 0:
                pl_masks = x['pl_masks']
                pl_masks = torch.stack(pl_masks, dim=1)
            else:
                pl_masks = None
            return self.forward_train(imgs, seq_ids, seq_names, paths, gt_fw_flows, gt_bw_flows, pl_masks)
        else:
            return self.forward_eval(imgs, seq_ids, seq_names, paths, return_pred_vis_list=return_pred_vis_list)
