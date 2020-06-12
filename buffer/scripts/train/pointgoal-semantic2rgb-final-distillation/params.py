import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [64, 128]:
    for resnet_model in ['se_resneXt50']:
        for lr in [1e-4, 1e-3]:
            for weight_decay in [5e-5]:
                job = f"""python -m buffer.train_rgb \\
--description {unique}-v5 \\
--max_epoch 100 \\
--checkpoint_dir /scratch/cluster/nimit/checkpoints \\
--source_teacher /scratch/cluster/nimit/models/apartment_0-semantic-teacher/model_latest.t7 \\
--goal_prediction /scratch/cluster/nimit/wandb/run-20200612_023226--1996829503092148540/model_010.t7 \\
--dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_2-k=3 \\
--resnet_model {resnet_model} \\
--hidden_size 256 \\
--batch_size {batch_size} \\
--lr {lr} \\
--weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
