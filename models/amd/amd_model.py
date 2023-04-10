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

from ..resnet import ResNet
from ..fcn_head import FCNHead

logger = utils.get_logger()

class AMDModel(nn.Module):
    """AMDModelv1: The original model that is compatible with the original code."""
    def __init__(self,
                 args,
                 backbone2,
                 decode_head,
                 decode_head2,
                 w_seg=2.0,
                 mask_layer=1,
                 train_iter=0,
                 train_cfg=None,
                 test_cfg=None,
                 log_interval=50):
        super(AMDModel, self).__init__()
        self.args = args
        self.save_dir = os.path.join(args.checkpoints_dir, "saved")
        self.save_dir_eval = os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_dir_name", "saved_eval"))
        self.save_dir_eval_export = os.path.join(args.checkpoints_dir, getattr(args, "saved_eval_export_dir_name", "saved_eval_export"))

        self.backbone2 = globals()[backbone2.pop('type')](**backbone2)
        
        self.mask_layer = mask_layer
        self.decode_head = self._init_decode_head(decode_head)
        self.decode_head2 = self._init_decode_head(decode_head2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

        self.train_iter = train_iter
        self.train_iter_per_log = 50

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.L1loss = nn.MSELoss()
        self.L1 = nn.L1Loss()

        self.has_dir = set([])

        self.w_seg = w_seg
        self.log_interval = log_interval

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        decode_head = globals()[
            decode_head.pop('type')](**decode_head)
        self.align_corners = decode_head.align_corners
        self.num_classes = decode_head.num_classes
        return decode_head

    def init_weights(self):
        self.backbone2.init_weights()

    def _decode_head_forward_with_img(self, x, imgs, decode_head, mask):
        """Run forward function and calculate loss for decode head in
        training."""
        pred_flows, loss_flow, flow_bg, flow_fg, flow_group = decode_head.flow_forward(
            imgs, mask)
        return -1, pred_flows, loss_flow, flow_bg, flow_fg, flow_group

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


    def forward_eval(self, imgs, seq_ids, seq_names, paths):
        losses = dict()
        _batch, im_num, _channel, _h, _w = imgs.shape
        img_3 = imgs.view(_batch * im_num, _channel, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(
            _batch, im_num, self.mask_layer, _feat_h, _feat_w)
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

        return pred_masks

    def forward_train(self, imgs, seq_ids, seq_names, paths):
        _batch, im_num, _channel, _h, _w = imgs.shape

        img_3 = imgs.view(_batch * im_num, _channel, _h, _w)
        all_feat = self.extract_feat(img_3, self.backbone2)
        all_pred_mask = self._decode_head_forward(all_feat, self.decode_head2)
        _, _, _feat_h, _feat_w = all_pred_mask.shape
        all_pred_mask = all_pred_mask.view(
            _batch, im_num, self.mask_layer, _feat_h, _feat_w)
        all_pred_mask = F.softmax(all_pred_mask, dim=2)

        _, pred_flows, loss_flow, flow_bg, flow_fg, flow_group = self._decode_head_forward_with_img(
            -1, imgs, self.decode_head, all_pred_mask)
        assert len(pred_flows['seg']) == 1 and len(pred_flows['whole']) == 1
        pred_flows_seg = [self.resize(it, all_pred_mask.shape[-2:])
                          for it in pred_flows['seg']][0]
        pred_flows_whole = [self.resize(
            it, all_pred_mask.shape[-2:]) for it in pred_flows['whole']][0]

        if self.train_iter % self.log_interval == 0:
            ith_imgs = []
            ith_flowxy_ori = []
            ith_flowxy_res = []

            mask_mat = [[all_pred_mask[:, it, iit:iit+1, :, :].repeat(
                1, 3, 1, 1) for it in range(im_num)] for iit in range(self.mask_layer)]
            pic_mask_mat = [torch.cat(it, 0) for it in mask_mat]

        loss_input_warp_seg = loss_flow['seg']

        line_mean_loss = 0.0
        for _i in range(im_num):
            ith_img = imgs[:, _i, :, :, :]
            ith_img = ith_img.detach()
            pred_mask = all_pred_mask[:, _i, :, :, :]

            if _i == 0:
                cur_flow_seg = pred_flows_seg[:, :2, :, :]
                cur_flow_whole = pred_flows_whole[:, :2, :, :]
            else:
                cur_flow_seg = pred_flows_seg[:, 2:, :, :]
                cur_flow_whole = pred_flows_whole[:, 2:, :, :]
                assert _i == 1

            if self.train_iter % self.log_interval == 0:
                for __i in range(_batch):
                    flow_color_3 = self.let_tensor_vis(cur_flow_seg[__i:__i+1])
                    flow_color_3_whole = self.let_tensor_vis(
                        cur_flow_whole[__i:__i+1])
                    flow_color_3 = torch.cat(
                        [flow_color_3, flow_color_3_whole], dim=2)
                    ith_flowxy_ori.append(flow_color_3 / 255.0)
                ith_img_resize = self.resize(ith_img, pred_mask.shape[-2:])
                ith_imgs.append((ith_img_resize+2)/4.0)

        if self.train_iter % self.log_interval == 0:
            # Filename uses idx_in_batch
            idx_in_batch = 0
            # 0 below means current_frame
            img_frame_id = paths[0][idx_in_batch].split('/')[-1][:-4]
            pic_ith_imgs = torch.cat(ith_imgs, 0)
            pic_ith_flowxy_ori = torch.cat(ith_flowxy_ori, 0)

            tosave = torch.cat(pic_mask_mat + [pic_ith_imgs, pic_ith_flowxy_ori], dim=2)

            try:
                fn_name = '{}/train_iter{:07d}_{}_{}_{}_img_pred_recons.jpg'.format(
                    self.save_dir, self.train_iter, seq_names[idx_in_batch], seq_ids[idx_in_batch], img_frame_id)
                torchvision.utils.save_image(tosave, fn_name)
            except:
                logger.warn(f"Error in saving: {fn_name}")
        self.train_iter += 1
        return loss_input_warp_seg * self.w_seg

    def forward(self, x):
        imgs, seq_ids, seq_names, paths = x['imgs'], x['seq_ids'], x['seq_names'], x['paths']
        # Stack images only (not annotations since they are not paired)
        imgs = torch.stack(imgs, dim=1)
        if self.training:
            return self.forward_train(imgs, seq_ids, seq_names, paths)
        else:
            return self.forward_eval(imgs, seq_ids, seq_names, paths)
