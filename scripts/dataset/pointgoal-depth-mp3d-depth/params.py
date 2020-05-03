import numpy as np
from pathlib import Path

jobs = list()

NUM_FRAMES = {
    'train': 10000,
    'val': 200,
    'test': 200
}

root = '/scratch/cluster/nimit/data/habitat/pointgoal-depth-mp3d-depth'

# train/test/val for MP3D, only depth
for split_f in Path('/u/nimit/Documents/robomaster/habitat2robomaster/splits').glob('mp3d_*'):
    if split_f.is_dir():
        continue

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
            --dataset mp3d \\
            --split "{split}" \\
            --scene {scene} \\
            --num_frames {NUM_FRAMES[split]} \\
            --shuffle \\
            --depth
        """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
