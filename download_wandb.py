import wandb
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project', '-p', type=str, required=True)
parser.add_argument('--models_root', '-r', type=Path, default=Path('models'))
parsed = parser.parse_args()

api = wandb.Api()

runs = api.runs(parsed.project)
for i, run in enumerate(runs):
    root = parsed.models_root / parsed.project / run.name
    root.mkdir(parents=True, exist_ok=True)

    if not (root / 'config.yaml').exists():
        run.file('config.yaml').download(root=root)

    models = [f for f in run.files() if 'model_' in f.name]
    if len(models) == 0:
        continue
    if models[-1].name.split('.')[0] not in [str(model.stem) for model in root.glob('model_*.t7')]:
        models[-1].download(root=root)

    print(f'[{i+1}/{len(runs)}] Downloaded {root}')
