import numpy as np
from pathlib import Path
import argparse

wandb_root = Path('/scratch/cluster/nimit/wandb')


jobs = list()
for run in wandb_root.glob('run-*'):
    for model in (run/'files').glob('model_*.t7'):
        epoch = int(model.stem.split('_')[1])
        job = f"""python -m again.evaluate \
    --model {model} \\
    --epoch {epoch} \\
    --split val \\
    --scene apartment_2
"""

        jobs.append(job)
        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
