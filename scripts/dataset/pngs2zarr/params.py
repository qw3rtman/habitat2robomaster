import numpy as np
from pathlib import Path
import datetime

jobs = list()

root = Path('/scratch/cluster/nimit/data/habitat/pointgoal-depth-gibson')

# train for Gibson, only RGB (since no semantic available)
for split_f in Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits').glob('gibson_*.txt'):
    split = split_f.stem.split('_')[1]
    with open(split_f, 'r') as f:
        scenes = [line.rstrip() for line in f]
    for scene in scenes:
        d = list(Path(root / f'{scene}-{split}').iterdir())[0]
        dt = datetime.datetime.fromtimestamp(d.stat().st_mtime)
        if dt.day == 30:
            continue

        print(dt)
        job = f"""python pngs2zarr.py \\
            --scene_dir {root}/{scene}-{split} \\
            --rgb
        """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
