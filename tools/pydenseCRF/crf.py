import argparse
import glob
import multiprocessing as mp
import tqdm
import os
from PIL import Image
import numpy as np
import torch

import pydensecrf.densecrf as dcrf
from scipy.ndimage import gaussian_filter


def get_parser():
    parser = argparse.ArgumentParser(description="CRF")
    parser.add_argument(
        "--input",
        help="Input directory",
    )
    parser.add_argument(
        "--output",
        help="Output directory",
    )
    parser.add_argument(
        "--annotation-dir",
        help="Annotation directory",
        default=None
    )
    parser.add_argument(
        "--allow_skip",
        action='store_true',
        help="Allow skipping if annotation does not exist"
    )
    parser.add_argument(
        "--step",
        default=0,
        type=int,
        help="The step of export, 0 by default (masks exported with export config will have 0 as step). Use 4320 for 20 epochs on davis."
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="*",
        help="sequence.",
    )
    return parser


color = np.array([0.0, 1.0, 1.0]).reshape((1, 1, 3))


def iou(a, b):
    i = a & b
    u = a | b
    return i.sum() / (u.sum() + 1e-10)


def refine(mask, image, gk, sxy, srgb, compat, gtmask):
    dfield = dcrf.DenseCRF2D(mask.shape[1], mask.shape[0], 2)

    U = gaussian_filter(mask, sigma=gk)
    U = U / (np.amax(U)+1e-8)
    U = np.clip(U, 1e-6, 1.0-1e-6)
    UU = np.zeros((2, mask.shape[0], mask.shape[1]))
    UU[1, :, :] = U
    UU[0, :, :] = 1.0-U
    UU = -np.log(UU)
    UU = np.float32(UU)
    UU = UU.reshape((2, -1))
    dfield.setUnaryEnergy(UU)

    im = np.ascontiguousarray(image).copy()
    dfield.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=im, compat=compat)
    Refine_num = 50
    Q = dfield.inference(Refine_num)
    new_mask = np.argmax(Q, axis=0).reshape((mask.shape[0], mask.shape[1]))
    new_mask = np.float32(new_mask)

    if gtmask is not None:
        gt = gtmask > 0.1
        bmask = new_mask > 0.1

        inter_new = gt & bmask
        union_new = gt | bmask
        iou_new = np.float32(np.sum(inter_new)) / np.float32(np.sum(union_new))

        return new_mask, iou_new

    return new_mask




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    created_dir = False

    args = get_parser().parse_args()

    visualize_only = first_frames_only = False
    val_seq_only = False

    val_seqs = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane',
                'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

    # allow_skip should be set to True for FBMS59
    allow_skip = args.allow_skip

    images_list = []
    anns_list = []
    ann_id = 0

    torch.set_grad_enabled(False)

    annotation_dir = args.annotation_dir

    assert annotation_dir is not None
    assert os.path.exists(annotation_dir)
    print("Annotation dir:", annotation_dir)

    skipped = 0

    seq = args.seq
    step = args.step

    if first_frames_only:
        paths = glob.glob(args.input + f'/{seq}/00000.jpg')
    else:
        paths = glob.glob(args.input + f'/{seq}/*.jpg')
    paths = sorted(paths)

    print("seq:", seq)
    print("len(paths):", len(paths))

    save_path_suffix = ""
    cur_paths = paths

    for path_idx, path in enumerate(tqdm.tqdm(cur_paths)):
        scene_name, frame_name = path.split("/")[-2:]

        # Sequences in validation set only
        if val_seq_only:
            if val_seqs is not None and scene_name not in val_seqs:
                continue

        annotation_img_path = f"{annotation_dir}/pred_seg_{scene_name}_{frame_name[:-4]}_{step:07}.png"
        if not allow_skip:
            assert os.path.exists(annotation_img_path), annotation_img_path
        else:
            if not os.path.exists(annotation_img_path):
                skipped += 1
                continue

        img_path = path

        # use PIL, to be consistent with evaluation
        img = np.asarray(Image.open(path))
        assert (img.shape[1], img.shape[0]) == (854, 480), img.shape

        img_np = img
        height, width, _ = img.shape

        mask = np.asarray(Image.open(annotation_img_path).resize(
            (img.shape[1], img.shape[0])))
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = (mask / 0.8).clip(min=0, max=255).astype(np.uint8)
        save_path = annotation_img_path.split("/")
        if len(save_path[-2]) > 1:
            save_path[-2] += "_crf" + save_path_suffix
        else:
            save_path[-3] += "_crf" + save_path_suffix
        save_path = "/".join(save_path)

        sxy = 25.
        srgb = 5.
        scomp = 5.
        gauss_k = 0.1
        # Use 480p
        sxy = 60.

        mask_new = refine(mask, img, gauss_k, sxy, srgb, scomp, gtmask=None)

        if not created_dir:
            # We need exist_ok=True when running multiple seqs
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            created_dir = True
        Image.fromarray((mask_new * 255.).astype(np.uint8)
                        ).convert('L').save(save_path)

    if skipped > 0:
        print(
            f"Skipped {skipped} frames (this number does not include the ones in training set if val_seq is True)")
