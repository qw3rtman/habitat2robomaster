import numpy as np
from pathlib import Path
import argparse

root = Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/content')

jobs = list()
for scene in root.iterdir():
    s = scene.stem.split('.')[0]
    job = f"""python -m again.collect_dataset \\
        --dataset gibson \\
        --scene {s} \\
        --split train \\
        --num_episodes 2000 \\
        --dataset_dir /scratch/cluster/nimit/data/habitat/gibson-{s}
    """

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
