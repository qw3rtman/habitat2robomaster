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
parser.add_argument('--dataset')
parser.add_argument('--scene')
parser.add_argument('--redo', action='store_true')
parser.add_argument('--split', type=str, default='val')
parsed = parser.parse_args()

api = wandb.Api()
run_dirs = list(checkpoint_root.glob(parsed.glob))
print(run_dirs)

runs = {}
if parsed.epoch is not None: # pick from wandb
    for run_dir in run_dirs:
        key = '-'.join(run_dir.name.split('-'))

        # get epoch from wandb
        run = api.runs(f'qw3rtman/pointgoal-depth2semantic-student/',
                {'config.run_name': key})[0]
        model = list(wandb_root.glob(f'*{run.id}'))[0]/f'model_{parsed.epoch:03}.t7'
        if model.exists():
            print(model)
            runs[key] = model, parsed.epoch
else: # use model_latest from checkpoints
    for run_dir in run_dirs:
        key = '-'.join(run_dir.name.split('-'))

        # get epoch from wandb
        run = api.runs(f'qw3rtman/pointgoal-depth2semantic-student/',
                {'config.run_name': key})[0]
        config_f = list(wandb_root.glob(f'*{run.id}'))[0]/'config.yaml'
        shutil.copy(config_f, run_dir/'config.yaml')

        epoch = int(run.summary['epoch'])
        run_name = f'{key}-model_{epoch:03}-{parsed.split}-new'
        if not parsed.redo:
            try:
                api.flush()
                run = api.run(f'qw3rtman/pointgoal-depth2semantic-eval/{run_name}')
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
    {f'--redo ' if parsed.redo else ''}{f'--dataset {parsed.dataset} ' if parsed.dataset else ''}{f'--scene {parsed.scene} ' if parsed.scene else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
