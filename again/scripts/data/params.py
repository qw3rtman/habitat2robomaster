import numpy as np
from pathlib import Path
import argparse

wandb_root = Path('/scratch/cluster/nimit/wandb')

jobs = list()
for scene in ['frl_apartment_4']:
    job = f"""python -m again.collect_dataset \\
        --dataset replica \\
        --scene {scene} \\
        --split train \\
        --num_episodes 100 \\
        --dataset_dir /scratch/cluster/nimit/data/habitat/replica-{scene}
    """

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
