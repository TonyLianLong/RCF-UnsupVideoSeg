import os
import numpy as np
from PIL import Image
import sys
from glob import glob

# This is the threshold above which we consider object
POS_TH = 0.35

class Results(object):
    def __init__(self, root_dir, step=0):
        self.root_dir = root_dir
        self.step = step

    def _read_mask(self, sequence, frame_id):
        try:
            if True:
                mask_path = os.path.join(self.root_dir, f'pred_seg_{sequence}_{frame_id}_{self.step:07}.png')
                if not os.path.exists(mask_path):
                    mask_path_query = os.path.join(self.root_dir, f'pred_seg_{sequence}_*_{frame_id}_{self.step:07}.png')
                    mask_path = glob(mask_path_query)
                    assert len(mask_path) == 1, mask_path_query
                    mask_path = mask_path[0]
            else:
                mask_path = os.path.join(self.root_dir, f'{sequence}/{frame_id}.png')
            arr = np.array(Image.open(mask_path).resize(size=(854, 480), resample=Image.BILINEAR))
            if arr.ndim == 3:
                arr = arr[..., 0]
            # print(arr.shape)
            return arr
        except IOError as err:
            sys.stdout.write(sequence + " frame %s not found!\n" % frame_id)
            sys.stdout.write("The frames have to be indexed PNG files placed inside the corespondent sequence "
                             "folder.\nThe indexes have to match with the initial frame.\n")
            sys.stderr.write("IOError: " + err.strerror + "\n")
            sys.exit()

    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = (self._read_mask(sequence, m)>256*POS_TH).astype(np.uint8)
        # print(masks.shape)
        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks
