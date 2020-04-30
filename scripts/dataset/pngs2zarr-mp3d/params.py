import numpy as np
from pathlib import Path

jobs = list()

root = '/scratch/cluster/nimit/data/habitat/pointgoal-depth-mp3d'

for split_f in Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits').glob('mp3d_*'):
    if split_f.is_dir():
        continue

    split = split_f.stem.split('_')[1]
    with open(split_f, 'r') as f:
        scenes = [line.rstrip() for line in f]
    for scene in scenes:
        job = f"""python pngs2zarr.py \\
            --scene_dir {root}/{scene}-{split} \\
            --rgb
        """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
