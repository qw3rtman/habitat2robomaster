import argparse
from pathlib import Path

root = Path('/scratch/cluster/nimit/data/habitat')

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=Path, required=True)
args = parser.parse_args()

s = args.scene_dir.stem.split('-')[1]
for split in args.scene_dir.iterdir():
    for ep in split.iterdir():
        (root/'testing'/split.stem/f'{s}-{ep.stem}').symlink_to(ep)
        #ep.rename(ep.parent/f'{ep.stem}-{len(list(ep.iterdir()))-1}')
