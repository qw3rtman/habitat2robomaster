import os

from pathlib import Path


SUBMIT = """Executable = run_{uuid}.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_{uuid}.log
Output=/u/nimit/logs/$(ClusterId)_{uuid}.out
Error=/u/nimit/logs/$(ClusterId)_{uuid}.err

Queue 1
"""


TRAIN = """#!/bin/zsh
source $HOME/.zshrc

export WANDB_DIR=/scratch/cluster/nimit
cd $HOME/Documents/robomaster/habitat2robomaster

conda env list
conda activate /scratch/cluster/nimit/miniconda3/envs/robo
"""

from params import JOBS


for job in JOBS:
    experiment = Path('.').resolve().name
    uuid = '{}_{}'.format(experiment, (len(list(Path('.').glob('*'))) - 1) // 2)
    print(uuid)

    Path('submit_%s.sh' % uuid).write_text(SUBMIT.format(uuid=uuid))
    Path('run_%s.sh' % uuid).write_text(TRAIN + job)

    os.chmod('run_%s.sh' % uuid, 509)

    print(uuid)
