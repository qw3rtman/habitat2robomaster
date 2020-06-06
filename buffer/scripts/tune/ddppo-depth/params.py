import numpy as np
from pathlib import Path
import argparse
import shutil
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--scene', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parsed = parser.parse_args()

jobs = list()
for mode in ['greedy', 'teacher']:
    for k in range(8):
        job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python buffer/visualize_expert.py \\
        --mode {mode} \\
        --dataset {parsed.dataset} \\
        --scene {parsed.scene} \\
        --split {parsed.split} \\
        --k {k}
    """

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
