import numpy as np
from pathlib import Path
import argparse

wandb_root = Path('/scratch/cluster/nimit/wandb')


jobs = list()
job = f"""python -m again.collect_dataset \\
    --dataset replica \\
    --scene apartment_0 \\
    --split train \\
    --num_episodes 10000 \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_0
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
