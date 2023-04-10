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


class NCutEvalHead(nn.Module):
    def __init__(self, args, resize_imgs_size=(480, 856), resize_masks_size=(480, 854),
                 arch="vit_small", patch_size=8, which_features="k", tau=0.2, eps=1e-5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.args = args

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

        ncut_value = soft_ncut_value(feats, masks, tau=self.tau, eps=self.eps)
        ncut_value = ncut_value[None]

        return ncut_value.cpu().numpy()


def get_pred(seq_name, frame_name, channel_ind, return_path=False):
    seq_ind = seq_name_to_ind_map[seq_name]

    if isinstance(frame_name, int):
        frame_name = f"{frame_name:05}"

    mask_path = f"{pred_masks_dir}/{channel_ind}/pred_seg_{seq_name}_{frame_name}_{step:07}.png"
    mask = Image.open(mask_path).resize((img_size[1], img_size[0]))
    # mask = TF.resize(mask, img_size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask = np.asarray(mask).astype(np.float32) / 255.

    if mask.ndim == 3:
        mask = mask[..., 0]

    if return_path:
        return mask, mask_path

    return mask


def get_gt(seq_name, frame_name):
    seq_ind = seq_name_to_ind_map[seq_name]

    if isinstance(frame_name, int):
        frame_name = f"{frame_name:05}"

    gt_path = f"{gt_dir}/{seq_name}/{frame_name}.png"
    if not os.path.exists(gt_path):
        return None
    gt = Image.open(gt_path).convert('L')

    return np.asarray(gt).astype(np.float32) / 255.


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


def ncut_seq(seq_name, object_channel, umi_th, allow_skipping_gt=False):
    frames = glob(f"{images_dir}/{seq_name}/*.jpg")
    frames.sort()

    ncut_values = []
    for frame_name in frames:
        frame_name = os.path.basename(frame_name)[:-4]

        if "fbms59" in dataset:
            gt = get_gt(seq_name, frame_name)
            if gt is None:  # No ground truth, skipping NCut
                assert allow_skipping_gt
                continue

        # print(frame_name)

        image = get_image(seq_name, frame_name)

        mask = get_pred(seq_name, frame_name, object_channel)

        image_tensor = torch.tensor(image, device=device)
        mask_tensor = torch.tensor(mask, device=device)

        ncut_value = ncut_head(
            image_tensor[None], mask_tensor[None], standardize=True)[0]

        ncut_values.append(ncut_value)

        if first_frames_only:
            break

    return ncut_values


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
    parser.add_argument('--step', type=int, default=0, help='The step of the export masks (should be 0 if exported with evaluation config)')

    args = parser.parse_args()

    dataset = args.dataset
    step = args.step

    print("Dataset:", dataset)

    pretrain_name = args.pretrain_dir
    object_channel = args.object_channel
    num_channels = args.num_channels
    first_frames_only = args.first_frames_only

    export_dir_name = 'saved_eval_export'

    # if dataset == "davis":
    #     export_dir_name = 'saved_eval_export_trainval'

    # elif dataset == "stv2":
    #     export_dir_name = 'saved_eval_export'

    # elif dataset == "fbms59":
    #     export_dir_name = 'saved_eval_export_trainval'

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

    gt_dir = images_dir.replace('JPEGImages', 'Annotations')

    seqs = sorted(os.listdir(images_dir))
    seqs = [seq for seq in seqs if not seq.startswith('.')]

    seq_name_to_ind_map = {seq: ind for ind, seq in enumerate(seqs)}
    print(f"Found {len(seqs)} sequences: {seqs}")

    img_size = (480, 854)

    ncut_head = NCutEvalHead(args=None)

    allow_skipping_gt = dataset == "fbms59"
    umi_th = 10000 if dataset == "fbms59" else None


    if object_channel is None:
        object_channel_options = list(range(num_channels))
    else:
        object_channel_options = [object_channel]

    frame_maas = []
    for object_channel in object_channel_options:
        dataset_maas = []

        for seq in tqdm(val_seqs):
            # NCut values
            current_seq_ncuts = ncut_seq(
                seq, object_channel=object_channel, umi_th=umi_th, allow_skipping_gt=allow_skipping_gt)
            current_seq_maas = -np.array(current_seq_ncuts)

            dataset_maas.extend(list(current_seq_maas))

        frame_maa = np.mean(dataset_maas)

        print(
            f"frame MAA with object channel {object_channel}: {frame_maa * 100.:.2f}")
        frame_maas.append(frame_maa)

    if len(object_channel_options) > 1:
        best_object_channel = int(np.argmax(np.array(frame_maas)))
        print(
            f"The best object channel among all channels evaluated is channel {best_object_channel}")
        
        sys.exit(best_object_channel)
