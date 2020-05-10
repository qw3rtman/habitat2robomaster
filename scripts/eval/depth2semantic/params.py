import numpy as np
from pathlib import Path

jobs = list()

models = [
    ('/scratch/cluster/nimit/wandb/run-20200504_052121-resnet50-bc-backprop-depth2semantic-mp3d-noaug-original-1.0-8-0.001-0.0005-v5.04-seg/model_010.t7', True)
]

for (model, compass) in models:
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python eval_hc.py \\
    --model {model} \\
    --split val \\
    {'--compass' if compass else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
