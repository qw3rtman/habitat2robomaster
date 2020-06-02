import numpy as np
from pathlib import Path
import argparse
import shutil
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--split', type=str, default='val')
parsed = parser.parse_args()

jobs = list()
for k in range(8):
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python buffer/tune_dagger.py \\
    --dataset {parsed.dataset} \\
    --split {parsed.split} \\
    --k {k}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
