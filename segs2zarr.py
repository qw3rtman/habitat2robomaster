import zarr
import os
import argparse
from habitat_dataset import HabitatDataset
from pathlib import Path
from numcodecs import Blosc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=Path, required=True)
parsed = parser.parse_args()

compressor = Blosc(cname='zstd', clevel=3)

for episode_dir in parsed.scene_dir.iterdir():
    if episode_dir.is_file():
        continue

    with open(episode_dir / 'episode.csv', 'r') as f:
        length = len(f.readlines()[1:])

    print(episode_dir)
    seg_seq = np.empty((length, 256, 256), dtype=np.uint8)
    for idx, seg_f in enumerate(episode_dir.glob('seg*')):
        seg_seq[idx, ...] = np.load(str(seg_f))['semantic']
        os.remove(seg_f)

    z = zarr.open(str(episode_dir / 'semantic'), mode='w', shape=(length, 256, 256), chunks=False, dtype='u1', compressor=compressor)
    z[:] = seg_seq
