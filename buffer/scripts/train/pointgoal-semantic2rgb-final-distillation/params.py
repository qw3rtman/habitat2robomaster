import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [64, 128]:
    for resnet_model in ['se_resneXt50']:
        for lr in [1e-4, 1e-3]:
            for weight_decay in [0.0]: #5e-5]:
                job = f"""wandb on; python -m buffer.train_rgb \\
--description {unique}-v4 \\
--max_epoch 100 \\
--checkpoint_dir /scratch/cluster/nimit/checkpoints \\
--source_teacher /scratch/cluster/nimit/models/apartment_0-semantic-teacher/002/model_030.t7 \\
--goal_prediction /scratch/cluster/nimit/checkpoints/resnet34_0.0001_5e-05_64_10.0_semantic_6.14-v1/model_latest.t7 \\
--dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_0-k=3-target \\
--resnet_model {resnet_model} \\
--hidden_size 512 \\
--batch_size {batch_size} \\
--lr {lr} \\
--weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
