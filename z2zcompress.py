import zarr
from numcodecs import Blosc
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=Path, required=True)
#parser.add_argument('--scene', required=True/
parser.add_argument('--rgb', action='store_true')
parser.add_argument('--semantic', action='store_true')
parsed = parser.parse_args()

compressor = Blosc(cname='zstd', clevel=3)

#scene_dir = list(parsed.dataset_dir.glob(f'{parsed.scene}*'))[0]
for episode_dir in parsed.scene_dir.iterdir():
    if episode_dir.is_file():
        continue

    print(episode_dir)
    if parsed.rgb:
        z = zarr.open(str(episode_dir / 'rgb'), mode='r')
        rgb = z[:]

        z = zarr.open(str(episode_dir / 'rgb'), mode='w', shape=rgb.shape, chunks=False, dtype='u1', compressor=compressor)
        z[:] = rgb
