import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()

jobs = list()
for model in args.root.glob('model_*.t7'):
    epoch = int(model.stem.split('_')[1])
    job = f"""python -m again.evaluate_aux \
--model {model} \\
--epoch {epoch} \\
--split val \\
--scene frl_apartment_4
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
