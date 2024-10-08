import numpy as np
from pathlib import Path

jobs = list()

NUM_FRAMES = {
    'train': 9350,
    'val': 270,
    'test': 270
}

root = '/scratch/cluster/nimit/data/habitat/pointgoal-depth-gibson'

# train for Gibson, only RGB (since no semantic available)
for split_f in Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits').glob('gibson_*.txt'):
    split = split_f.stem.split('_')[1]
    with open(split_f, 'r') as f:
        scenes = [line.rstrip() for line in f]
    for scene in scenes:
        if Path(f'{root}/{scene}-{split}').exists():
            continue

        job = f"""python habitat_wrapper.py \\
            --task pointgoal \\
            --mode teacher \\
            --proxy depth \\
            --dataset_dir {root}/{scene}-{split} \\
            --num_episodes 50 \\
            --dataset gibson \\
            --split "{split}_ddppo" \\
            --scene {scene} \\
            --num_frames {NUM_FRAMES[split]} \\
            --shuffle \\
            --rgb
        """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
