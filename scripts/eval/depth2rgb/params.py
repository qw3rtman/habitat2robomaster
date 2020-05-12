import numpy as np
from pathlib import Path
import argparse
import os


root = Path('/scratch/cluster/nimit/wandb')

parser = argparse.ArgumentParser()
parser.add_argument('--glob', type=str, required=True)
parser.add_argument('--compass', action='store_true')
parsed = parser.parse_args()

runs = {}
run_dirs = list(root.glob(parsed.glob))
run_dirs.sort(key=os.path.getmtime)
for run_dir in run_dirs:
    key = '-'.join(run_dir.stem.split('-')[2:])
    models = list(run_dir.glob('model_*.t7'))
    models.sort(key=os.path.getmtime)

    if len(models) == 0:
        continue
    model = models[-1]
    epoch = int(model.stem.split('_')[1].split('.')[0])

    runs[key] = model, epoch

jobs = list()
for (model, epoch) in runs.values():
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python eval_hc.py \\
    --model {model} \\
    --epoch {epoch} \\
    --split val \\
    {'--compass' if parsed.compass else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
