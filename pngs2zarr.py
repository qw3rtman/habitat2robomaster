import zarr
import os
import cv2
import argparse
from habitat_dataset import HabitatDataset
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=Path, required=True)
#parser.add_argument('--scene', required=True/
parser.add_argument('--rgb', action='store_true')
parser.add_argument('--semantic', action='store_true')
parsed = parser.parse_args()

#scene_dir = list(parsed.dataset_dir.glob(f'{parsed.scene}*'))[0]
for episode_dir in parsed.scene_dir.iterdir():
    if episode_dir.is_file():
        continue

    print(episode_dir)
    episode = HabitatDataset(episode_dir, rgb=parsed.rgb, semantic=parsed.semantic)

    if parsed.rgb:
        rgb_seq = np.empty((len(episode), 256, 256, 3), dtype=np.uint8)
        for idx, img_f in enumerate(episode.imgs):
            rgb_seq[idx, ...] = cv2.imread(img_f)
            os.remove(img_f)

        z = zarr.open(str(episode_dir / 'rgb'), mode='w', shape=(len(episode), 256, 256, 3), chunks=False, dtype='u1')
        z[:] = rgb_seq

    if parsed.semantic:
        pass
