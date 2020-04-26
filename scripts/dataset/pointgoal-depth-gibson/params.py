import numpy as np
from pathlib import Path

jobs = list()

for split_f in Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits').glob('gibson_*'):
    split = split_f.stem.split('_')[1]
    with open(split_f, 'r') as f:
        scenes = [line.rstrip() for line in f]
    for scene in scenes:
        job = f"""python habitat_wrapper.py \\
            --task pointgoal \\
            --mode teacher \\
            --proxy depth \\
            --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-gibson/{scene}-{split} \\
            --num_episodes 50 \\
            --dataset gibson \\
            --split {split} \\
            --scene {scene} \\
            --num_frames 10500
        """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
