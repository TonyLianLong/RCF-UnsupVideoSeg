import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from models.crf_head import CRFHead
from models.dino_vit import get_dino_model

device = "cuda"


def soft_ncut_value(feats, mask, tau, eps):
    feats = feats[0, 1:, :]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1, 0))

    A = A > tau

    A = torch.where(A == 0, eps, A)

    x = mask.view((-1,)).float()

    # A: x, B: 1-x
    cutAB = (1-x) @ (A @ x)
    assocAV = torch.sum(A @ x)
    assocBV = torch.sum(A @ (1-x))
    NCut = cutAB / assocAV + cutAB / assocBV

    return NCut


def ncut_refine(feats, masks, tau=0.2, eps=1e-5, steps=10, learning_rate=1e-1, weight_decay=1e-6,
                visualize_interval=10, visualize=False):
    masks = masks.clone()
    masks.requires_grad = True

    optimizer = torch.optim.Adam(
        [masks],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for i in range(steps):
        masks_soft_ncut_values = soft_ncut_value(
            feats, mask=masks, tau=tau, eps=eps)

        loss = masks_soft_ncut_values.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # This is only needed if masks are unconstrained (can be above 1 or below 0)
        with torch.no_grad():
            masks[:] = torch.clamp(masks, min=0., max=1.)

        masks_soft_ncut_values_item = masks_soft_ncut_values.item()

        if visualize and (i % visualize_interval == 0 or i == steps-1):
            print(masks_soft_ncut_values_item)
            plt.imshow(masks.detach().cpu().numpy())
            plt.title(f"Step: {i}")
            plt.axis("off")
            plt.show()

    return masks.detach()


class NCutHead(nn.Module):
    def __init__(self, args, steps=10, learning_rate=1e-1, weight_decay=1e-6, visualize_interval=10, visualize=False, resize_imgs_size=(480, 856), resize_masks_size=(480, 854),
                 arch="vit_small", patch_size=8, which_features="k", tau=0.2, eps=1e-5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.args = args

        self.steps = steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.visualize_interval = visualize_interval
        self.visualize = visualize

        self.mean = torch.tensor(mean, dtype=torch.float, device="cuda")[
            None, :, None, None]
        self.std = torch.tensor(std, dtype=torch.float, device="cuda")[
            None, :, None, None]

        self.resize_imgs_size = resize_imgs_size
        self.resize_masks_size = resize_masks_size

        self.arch = arch
        self.patch_size = patch_size

        self.which_features = which_features

        self.tau = tau
        self.eps = eps

        assert "vit" in self.arch, self.arch

        self.model = get_dino_model(self.arch, self.patch_size, device="cuda")

        for param in self.model.parameters():
            param.requires_grad = False

        # Store the outputs of qkv layer from the last attention layer
        self.feat_out = {}

        def hook_fn_forward_qkv(module, input, output):
            self.feat_out["qkv"] = output
        self.model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
            hook_fn_forward_qkv)

        self.h_featuremap = self.resize_imgs_size[0] // self.patch_size
        self.w_featuremap = self.resize_imgs_size[1] // self.patch_size

    def normalize(self, imgs):
        # B, H, W, 3 to B, 3, H, W
        imgs = imgs.permute((0, 3, 1, 2))
        imgs = (imgs - self.mean) / self.std

        return imgs

    def get_feats(self, imgs):
        self.feat_out = {}

        # Forward pass in the model
        attentions = self.model.get_last_selfattention(imgs)

        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens

        # Extract the qkv features of the last attention layer
        qkv = (
            self.feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )

        # Modality selection
        if self.which_features == "k":
            k = qkv[1]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            #feats = k[:, 1:, :]
            feats = k
        elif self.which_features == "q":
            q = qkv[0]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            #feats = q[:, 1:, :]
            feats = q
        elif self.which_features == "v":
            v = qkv[2]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            #feats = v[:, 1:, :]
            feats = v

        return feats

    def forward(self, imgs, masks, standardize=False):
        if standardize:
            imgs = self.normalize(imgs)

        B, _, H, W = imgs.shape

        imgs = F.interpolate(imgs, self.resize_imgs_size, mode='bilinear')
        masks = F.interpolate(
            masks[:, None, ...], (self.h_featuremap, self.w_featuremap), mode='nearest')[:, 0, ...]

        feats = self.get_feats(imgs)

        refined_mask = ncut_refine(feats, masks, tau=self.tau, eps=self.eps, steps=self.steps,
                                   learning_rate=self.learning_rate, weight_decay=self.weight_decay, visualize_interval=self.visualize_interval, visualize=self.visualize)

        refined_mask = F.interpolate(
            refined_mask[:, None, ...], self.resize_masks_size, mode='bilinear')

        refined_mask = refined_mask[:, 0, ...]

        return refined_mask.float()


def get_pred(seq_name, frame_name, channel_ind, return_path=False):
    seq_ind = seq_name_to_ind_map[seq_name]

    if isinstance(frame_name, int):
        frame_name = f"{frame_name:05}"

    mask_path = f"{pred_masks_dir}/{channel_ind}/pred_seg_{seq_name}_{frame_name}_0000000.png"
    mask = Image.open(mask_path).resize((img_size[1], img_size[0]))
    # mask = TF.resize(mask, img_size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask = np.asarray(mask).astype(np.float32) / 255.

    if mask.ndim == 3:
        mask = mask[..., 0]

    if return_path:
        return mask, mask_path

    return mask


def get_image(seq_name, frame_name):
    if isinstance(frame_name, int):
        frame_name = f"{frame_name:05}"

    img_path = f"{images_dir}/{seq_name}/{frame_name}.jpg"
    img = Image.open(img_path).convert('RGB')

    # img = TF.resize(img, img_size, interpolation=transforms.InterpolationMode.BICUBIC)
    img = np.asarray(img).astype(np.float32) / 255.

    assert img.shape == (480, 854, 3)

    return img


def visualize_masks(masks):
    plt.figure(figsize=(18, 8))
    for i, mask in enumerate(masks):
        plt.subplot(1, num_channels, i+1)
        plt.imshow(mask, vmin=0., vmax=1., cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_image_masks(image, masks):
    plt.figure(figsize=(18, 8))
    plt.subplot(1, num_channels+1, 1)
    plt.imshow(image)
    plt.axis("off")
    for i, mask in enumerate(masks):
        plt.subplot(1, num_channels+1, i+2)
        plt.imshow(mask, vmin=0., vmax=1., cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_image_mask(image, mask, cmap="gray"):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, vmin=0., vmax=1., cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def iou(a, b):
    i = a & b
    u = a | b
    if u.sum() == 0:
        return float('nan')
    return i.sum() / u.sum()


def umi(a, b):
    i = a & b
    u = a | b
    if u.sum() == 0:
        return float('nan')

    return u.sum() - i.sum()


def ncut_seq_and_save(seq_name, object_channel, umi_th, dry_run=False):
    frames = glob(f"{images_dir}/{seq_name}/*.jpg")
    frames.sort()

    for frame_path in frames:
        frame_name = os.path.basename(frame_path)[:-4]

        # print(frame_name)

        image = get_image(seq_name, frame_name)

        mask, mask_path = get_pred(
            seq_name, frame_name, object_channel, return_path=True)

        image_tensor = torch.tensor(image, device=device)
        mask_tensor = torch.tensor(mask, device=device)

        # refined_mask_tensor = crf_head(image_tensor[None], mask_tensor[None], unstandardize=False)[0]

        if ncut_with_crf:
            # With CRF
            # refined_mask_tensor = ncut_head(image_tensor[None], mask_tensor[None], standardize=True)
            # refined_mask_tensor = crf_head(image_tensor[None], refined_mask_tensor, unstandardize=False)[0]

            # With CRF, NCut CRF, and merge
            mask_tensor_expanded = mask_tensor[None]
            crf_refined_mask_tensor = crf_head_single(
                image_tensor[None], mask_tensor_expanded, unstandardize=False)[0]

            refined_mask_tensor = ncut_head(
                image_tensor[None], mask_tensor_expanded, standardize=True)
            refined_mask_tensor = crf_head(
                image_tensor[None], refined_mask_tensor, unstandardize=False)[0]

            if umi_th is not None:
                umi_frame = umi(crf_refined_mask_tensor.cpu().numpy(
                ) > 0.5, refined_mask_tensor.cpu().numpy() > 0.5)
                if umi_frame > umi_th:
                    # Skip refinement: likely captures different things
                    refined_mask_tensor = crf_refined_mask_tensor
                    # print(f"Skipping {seq_name}, {frame_name}")
                else:
                    refined_mask_tensor = crf_refined_mask_tensor * refined_mask_tensor
            else:
                refined_mask_tensor = crf_refined_mask_tensor * refined_mask_tensor

        else:
            refined_mask_tensor = ncut_head(
                image_tensor[None], mask_tensor[None], standardize=True)[0]

        refined_mask_np = refined_mask_tensor.cpu().numpy()

        save_mask_path = mask_path.replace(
            export_dir_name, export_dir_name + save_suffix)
        assert not os.path.exists(save_mask_path), save_mask_path
        refined_mask_pil = Image.fromarray(
            (refined_mask_np * 255.).astype(np.uint8)).convert('L')
        if dry_run:
            break
        else:
            refined_mask_pil.save(save_mask_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluate motion-appearance alignment.')
    parser.add_argument('--pretrain_dir', help='path to pretraining dir',
                        default=None, type=str)
    parser.add_argument('--first-frames-only', help='override pretrained model and checkpoints directory at test',
                        action='store_true')
    parser.add_argument('--num-channels', default=4, type=int)
    parser.add_argument('--object-channel', default=None, type=int,
                        help='object channel, if not supplied, perform MAA on all object channels and select the one with best MAA')
    parser.add_argument('--dataset', type=str, help='dataset', default="davis",
                        choices=["davis", "stv2", "fbms59"])

    args = parser.parse_args()

    dataset = args.dataset

    print("Dataset:", dataset)

    pretrain_name = args.pretrain_dir
    object_channel = args.object_channel
    num_channels = args.num_channels
    first_frames_only = args.first_frames_only

    save_suffix = "_torchcrf_ncut_torchcrf"

    if dataset == "davis":
        export_dir_name = 'saved_eval_export_trainval_ema'

    elif dataset == "stv2":
        export_dir_name = 'saved_eval_export_ema'

    elif dataset == "fbms59":
        export_dir_name = 'saved_eval_export_trainval_ema'

    pred_masks_dir = f"{pretrain_name}/{export_dir_name}"

    data_dir = "data"

    if dataset == "davis":
        data_root = f"{data_dir}/data_davis"
        images_dir = f"{data_root}/JPEGImages/480p"

        val_seqs = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
                    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

    elif dataset == "stv2":
        data_root = f'{data_dir}/data_SegTrackv2_resized'
        images_dir = f"{data_root}/JPEGImages"

        # All sequences
        val_seqs = [
            'bird_of_paradise',
            'birdfall',
            'bmx',
            'cheetah',
            'drift',
            'frog',
            'girl',
            'hummingbird',
            'monkey',
            'monkeydog',
            'parachute',
            'penguin',
            'soldier',
            'worm'
        ]

    elif dataset == "fbms59":
        # This is FBMS59_resized.
        data_root = f"{data_dir}/data_fbms59_resized"
        # Validation sequences
        val_seqs = ['camel01', 'cars1', 'cars10', 'cars4', 'cars5', 'cats01', 'cats03', 'cats06',
                    'dogs01', 'dogs02', 'farm01', 'giraffes01', 'goats01', 'horses02', 'horses04',
                    'horses05', 'lion01', 'marple12', 'marple2', 'marple4', 'marple6', 'marple7', 'marple9',
                    'people03', 'people1', 'people2', 'rabbits02', 'rabbits03', 'rabbits04', 'tennis']
        images_dir = f"{data_root}/JPEGImages"


    seqs = sorted(os.listdir(images_dir))
    seqs = [seq for seq in seqs if not seq.startswith('.')]

    seq_name_to_ind_map = {seq: ind for ind, seq in enumerate(seqs)}
    print(f"Found {len(seqs)} sequences: {seqs}")

    img_size = (480, 854)

    ncut_head = NCutHead(args=None, steps=10, learning_rate=0.45,
                         weight_decay=1e-6, visualize_interval=10, visualize=False)

    # args is unused in CRFHead for now
    # crf_head_single is for crf on the image only, crf_head is for crf after ncut optimization
    crf_head_single = CRFHead(args=None, srgb=5., scomp=5., sxy=60.,
                              scomp_smooth=0., sxy_smooth=0., refine_iters=50, crf_scale=0.7)
    crf_head = CRFHead(args=None, srgb=5., scomp=5., sxy=60.,
                       scomp_smooth=0., sxy_smooth=0., refine_iters=50, crf_scale=0.5)

    umi_th = 10000 if dataset == "fbms59" else None

    ncut_with_crf = True
    crf_and_save = True
    # trainval or val
    crf_and_save_trainval = True

    if crf_and_save_trainval:
        crf_seqs = seqs
    else:
        crf_seqs = val_seqs

    save_path = f"{pred_masks_dir}/{object_channel}".replace(
        export_dir_name, export_dir_name + save_suffix)
    os.makedirs(save_path, exist_ok=True)

    print(f"Start refinement: {save_path}")

    for seq in tqdm(crf_seqs):
        ncut_seq_and_save(seq, umi_th=umi_th, object_channel=object_channel)
