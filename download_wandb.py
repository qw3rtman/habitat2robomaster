import wandb
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', '-p', type=str, required=True)
parser.add_argument('--models_root', '-r', type=Path, default=Path('models'))
parsed = parser.parse_args()

api = wandb.Api()

runs = api.runs(project)
for run in runs:
    root = parsed.models_root / parsed.project / run.name
    root.mkdir(parents=True, exist_ok=True)

    if not (root / 'config.yaml').exists():
        run.file('config.yaml').download(root=root)

    model = [f for f in run.files() if 'model_' in f.name][-1]
    if model not in list(root.glob('model_*.t7')):
        model.download(root=root)
