import numpy as np
from pathlib import Path
import argparse
import os


root = Path('/scratch/cluster/nimit/wandb')

parser = argparse.ArgumentParser()
parser.add_argument('--glob', type=str, required=True)
parser.add_argument('--compass', action='store_true')
parsed = parser.parse_args()

runs = []
for run_dir in root.glob(parsed.glob):
    key = '-'.join(run_dir.stem.split('-')[2:])
    models = list(run_dir.glob('model_*.t7'))
    models.sort(key=os.path.getmtime)
    runs.append((models[-1], parsed.compass))

jobs = list()
for (model, compass) in runs:
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python eval_hc.py \\
    --model {model} \\
    --split val \\
    {'--compass' if compass else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
