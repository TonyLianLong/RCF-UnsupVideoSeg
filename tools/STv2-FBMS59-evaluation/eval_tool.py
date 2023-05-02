# This is for evaluating exported images. We first scale the exported image to the size of annotation and then calculate the IoU.

import os
from PIL import Image
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Train segmentation.')
parser.add_argument('--dataset', choices=['SegTrackv2', 'FBMS59'],
                    help='Dataset: either SegTrackv2 (resized training output) or FBMS59 (resized training output)')
parser.add_argument('--step', type=int, default=0, help="The step to evaluate, should use 0 if you export with the export config.")
parser.add_argument('--pred_dir', type=str,
                    help='Prediction directory (the directory which includes the prediction masks directly, often with the name of a channel index)')

args = parser.parse_args()

pred_dir = args.pred_dir

if args.dataset == "SegTrackv2":
    data_root = "data/data_SegTrackv2"
    dataset_file = "data/data_SegTrackv2/trainval.txt"

    # True for STv2 without resize since it does not rename
    use_original_filename = False
    allow_skipping_gt = False
    use_png = False
elif args.dataset == "FBMS59":
    data_root = "data/data_fbms59"

    # Use val since we need to ensure the filename is right (need the full sequence)
    dataset_file = "data/data_fbms59/val_all.txt"

    # Using val so we need to skip
    allow_skipping_gt = True
    use_original_filename = False

    # Use png as the ground truth file extension
    use_png = True

POS_TH = 0.35

with open(dataset_file, 'r') as f:
    seqs = f.readlines()

def iou(a, b):
    i = a & b
    u = a | b
    if u.sum() == 0:
        return float('nan')
    return i.sum() / u.sum()

all_ious = []

step = args.step

for seq in seqs:
    seq = seq.rstrip()
    seq = seq.split()
    seq_dir = seq[0].replace("JPEGImages", "Annotations")
    seq_name = seq_dir.split("/")[-2]
    frames = seq[1:]
    
    seq_ious = []
    
    for frame_ind, frame in enumerate(frames):
        path = os.path.join(data_root, seq_dir, frame)
        if use_png:
            path = path.replace(".jpg", ".png")
        if not os.path.exists(path):
            assert allow_skipping_gt, f"{path} does not exist, but skipping ground truth is not allowed"
            continue
        annotation = np.array(Image.open(path)) / 255.
        if annotation.ndim == 3:
            annotation = annotation[..., 0]
        
        if use_original_filename:
            frame_name = frame[:-4]
        else:
            frame_name = f"{frame_ind:05}"

        pred_path = os.path.join(pred_dir, f"pred_seg_{seq_name}_{frame_name}_{step:07}.png")
        pred = np.array(Image.open(pred_path).resize((annotation.shape[1], annotation.shape[0]))) / 255.
        if pred.ndim == 3:
            pred = pred[..., 0]
        
        # if len(np.unique(annotation)) > 2:
        #     print(np.unique(annotation, return_counts=True))
        
        annotation = annotation > 0.5
        pred = pred > POS_TH

        assert annotation.shape == pred.shape, f"{annotation.shape} != {pred.shape}"

        pred_iou = iou(pred, annotation)
        seq_ious.append(pred_iou)

    seq_iou = np.nanmean(seq_ious)
    all_ious += seq_ious

    print(f"mIoU on {seq_name}: {seq_iou * 100:.2f}")

all_iou = np.nanmean(all_ious)

print(f"mIoU: {all_iou * 100:.2f}")
print(f"Number of frames: {len(all_ious)}")
