import numpy as np
from pathlib import Path
import argparse
import shutil
import wandb
import os

wandb_root = Path('/scratch/cluster/nimit/wandb')
checkpoint_root = Path('/scratch/cluster/nimit/checkpoints')

parser = argparse.ArgumentParser()
parser.add_argument('--glob', type=str, required=True)
parser.add_argument('--epoch', type=int)
parser.add_argument('--redo', action='store_true')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--compass', action='store_true')
parsed = parser.parse_args()

runs = {}
if parsed.epoch is not None: # pick from wandb
    run_dirs = list(wandb_root.glob(parsed.glob))
    run_dirs.sort(key=os.path.getmtime)
    for run_dir in run_dirs:
        key = '-'.join(run_dir.stem.split('-')[2:])
        model = run_dir / f'model_{parsed.epoch:03}.t7'
        if model.exists():
            runs[key] = model, parsed.epoch
else: # use model_latest from checkpoints
    api = wandb.Api()
    run_dirs = list(checkpoint_root.glob(parsed.glob))
    for run_dir in run_dirs:
        key = '-'.join(run_dir.name.split('-'))

        # copy config.yaml
        wandb_dirs = list(wandb_root.glob(f'*{key}'))
        wandb_dirs.sort(key=os.path.getmtime)

        i = -1
        while not (wandb_dirs[i]/'config.yaml').exists():
            i -= 1
        shutil.copy(wandb_dirs[i]/'config.yaml', run_dir/'config.yaml')

        # get epoch from wandb
        run = api.run(f'qw3rtman/habitat-pointgoal-depth-student/{key}')
        epoch = 10*(run.summary['epoch']//10) # floor to nearest 10

        run_name = f'{key}-model_{epoch:03}-{parsed.split}-new'
        if not parsed.redo:
            try:
                api.flush()
                run = api.run(f'qw3rtman/pointgoal-rgb2depth-eval-hc/{run_name}')
                continue
            except:
                pass

        model = run_dir / f'model_latest.t7'
        runs[key] = model, epoch

jobs = list()
for (model, epoch) in runs.values():
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python buffer/eval_hc.py \\
    --model {model} \\
    --epoch {epoch} \\
    --split {parsed.split} \\
    {'--compass' if parsed.compass else ''} {'--redo ' if parsed.redo else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
