import numpy as np
from pathlib import Path
import argparse

wandb_root = Path('/scratch/cluster/nimit/wandb')

parser = argparse.ArgumentParser()
parser.add_argument('--glob', required=True)
parsed = parser.parse_args()

jobs = list()
for run in wandb_root.glob(parsed.glob):
    for model in (run/'files').glob('model_*.t7'):
        epoch = int(model.stem.split('_')[1])
        job = f"""python -m again.evaluate \
    --model {model} \\
    --epoch {epoch} \\
    --split val \\
    --scene apartment_0
"""

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
