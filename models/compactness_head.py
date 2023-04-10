import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactnessHead(nn.Module):
    # The compactness loss in GWM paper (v1).

    def __init__(self, args, compact_channel):
        super().__init__()
        
        self.args = args
        self.compact_channel = compact_channel
    
    def get_compactness_loss(self, all_pred_mask):
        all_pred_mask = all_pred_mask.flatten(0, 1)
        # all_pred_mask (after flatten): [B * I, C=5, 48, 48] (range from 0 to 1)
        # pred_mask_compact_channel: [B * I, 48, 48]
        
        if self.compact_channel == -1:
            # Use object channel as compact channel
            if self.args.object_channel is None:
                return None
            else:
                compact_channel = self.args.object_channel
        else:
            # Use pre-set compact channel
            compact_channel = self.compact_channel

        pred_mask_compact_channel = all_pred_mask[:, compact_channel]
        _, mask_H, mask_W = pred_mask_compact_channel.shape
        
        # pixel_count is soft: [B * I, 1, 1]
        pixel_count = pred_mask_compact_channel.sum(dim=(1, 2), keepdim=True)
        
        # Use arange instead of linspace because we don't want to include the endpoint (the max is slightly less than 1)
        # [1, mask_H, 1]
        y_loc = torch.arange(mask_H, dtype=torch.float, device=all_pred_mask.device)[None, :, None] / mask_H
        # [1, 1, mask_W]
        x_loc = torch.arange(mask_W, dtype=torch.float, device=all_pred_mask.device)[None, None, :] / mask_W
        
        # y_center, x_center: [B * I, 1, 1]
        y_center = (y_loc * pred_mask_compact_channel).sum(dim=(1, 2), keepdim=True) / pixel_count
        x_center = (x_loc * pred_mask_compact_channel).sum(dim=(1, 2), keepdim=True) / pixel_count
        
        # If you put mask here, your error will be the value in the error map
        # error_map: [B * I, mask_H=48, mask_W=48]
        error_map = (y_loc - y_center) ** 2 + (x_loc - x_center) ** 2
        
        # For debugging error_map:
        # import matplotlib.pyplot as plt
        # x_center[5] = 12/48.
        # y_center[5] = 38/48.
        # error_map = (y_loc - y_center) ** 2 + (x_loc - x_center) ** 2
        # [(plt.imshow(item), plt.savefig(f"error_map_{idx}.jpg")) for idx, item in enumerate(error_map.detach().cpu().numpy())]
        
        loss = (error_map * pred_mask_compact_channel).mean()
        return loss
