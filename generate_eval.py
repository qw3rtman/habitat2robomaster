from pathlib import Path
import argparse
import os

root = Path('/scratch/cluster/nimit/wandb')

parser = argparse.ArgumentParser()
parser.add_argument('--glob', type=str, required=True)
parsed = parser.parse_args()

for run_dir in root.glob(parsed.glob):
    key = '-'.join(run_dir.stem.split('-')[2:])
    models = list(run_dir.glob('model_*.t7'))
    models.sort(key=os.path.getmtime)
    models[-1]
