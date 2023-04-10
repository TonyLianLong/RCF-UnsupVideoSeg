import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchcrf_cpp
import pydensecrf.densecrf as dcrf

# Reference: https://github.com/antonilo/unsupervised_detection


class CRFHead(nn.Module):
    def __init__(self, args, srgb=5., scomp=5., sxy=60., scomp_smooth=0., sxy_smooth=0., refine_iters=50, crf_scale=0.7,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.args = args

        self.srgb = srgb
        self.scomp = scomp
        self.sxy = sxy

        self.crf_scale = crf_scale

        self.scomp_smooth = scomp_smooth
        self.sxy_smooth = sxy_smooth
        self.refine_iters = refine_iters

        self.mean = torch.tensor(mean, dtype=torch.float, device="cuda")[
            None, :, None, None]
        self.std = torch.tensor(std, dtype=torch.float, device="cuda")[
            None, :, None, None]

    def unnormalize(self, imgs):
        # B, 3, H, W to B, H, W, 3
        imgs = imgs * self.std + self.mean
        imgs = imgs.permute((0, 2, 3, 1))
        return imgs

    def crf(self, img, mask):
        img = img.contiguous()
        H, W, _ = img.shape

        mask = (mask * 255. / self.crf_scale).clip(min=0,
                                                   max=255).type(torch.uint8)

        U = mask
        U = U / (torch.max(U) + 1e-8)
        U = torch.clamp(U, 1e-6, 1.0-1e-6)
        UU = torch.zeros((2, *mask.shape), device="cuda")
        UU[1, :, :] = U
        UU[0, :, :] = 1.0-U
        UU = -torch.log(UU)
        # The memory order is different from pydensecrf library
        UU = UU.view((2, -1)).T
        UU = UU.contiguous()

        refined_mask = torchcrf_cpp.crf_soft(
            img, UU, W, H, self.scomp_smooth, self.sxy_smooth, self.scomp, self.sxy, self.srgb, self.refine_iters)

        return refined_mask.float()

    def crf_cpu(self, img, mask):
        image = img.contiguous().cpu().numpy()
        mask = mask.cpu().numpy()
        # The scale used by cpu is 0.8 in the original implementation.
        mask = (mask * 255. / self.crf_scale).clip(min=0,
                                                   max=255).astype(np.uint8)
        dfield = dcrf.DenseCRF2D(mask.shape[1], mask.shape[0], 2)

        U = mask
        # U = gaussian_filter(mask, sigma=gk)

        U = U / (np.amax(U)+1e-8)
        U = np.clip(U, 1e-6, 1.0-1e-6)
        UU = np.zeros((2, mask.shape[0], mask.shape[1]))
        UU[1, :, :] = U
        UU[0, :, :] = 1.0-U
        UU = -np.log(UU)
        UU = np.float32(UU)
        UU = UU.reshape((2, -1))
        dfield.setUnaryEnergy(UU)

        im = np.ascontiguousarray(image)
        dfield.addPairwiseBilateral(
            sxy=self.sxy, srgb=self.srgb, rgbim=im, compat=self.scomp)

        Q = dfield.inference(self.refine_iters)
        new_mask = np.argmax(Q, axis=0).reshape((mask.shape[0], mask.shape[1]))
        new_mask = np.float32(new_mask)

        return torch.tensor(new_mask, dtype=torch.float, device="cuda")

    def forward(self, imgs, masks, unstandardize=True):
        if unstandardize:
            imgs = self.unnormalize(imgs)

        imgs = imgs * 255.
        imgs = imgs.clamp_(min=0., max=255.).type(torch.uint8)

        refined_masks = []

        for img, mask in zip(imgs, masks):
            refined_mask = self.crf(img, mask)
            # refined_mask = self.crf_cpu(img, mask)
            refined_masks.append(refined_mask)

        refined_masks = torch.stack(refined_masks, dim=0)
        # from PIL import Image; Image.fromarray(imgs[0].cpu().numpy()).save("visualize.png"); Image.fromarray((masks[0].detach().cpu().numpy() * 255.).astype(np.uint8)).save("visualize_original_mask.png"); Image.fromarray((refined_masks[0].cpu().numpy() * 255.).astype(np.uint8)).save("visualize_mask.png")
        return refined_masks
