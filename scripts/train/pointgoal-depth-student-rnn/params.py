import numpy as np
from pathlib import Path

jobs = list()

for resnet_model in ['resnet50', 'se_resneXt50']:
    for batch_size in [8, 16]:
        for lr in [1e-3, 1e-4]:
            for weight_decay in [5e-4, 5e-5]:
                job = f"""python train_pointgoal_student_rnn.py \\
    --max_epoch 100 \\
    --resnet_model {resnet_model} \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-mp3d \\
    --teacher_task pointgoal \\
    --teacher_proxy depth \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --weight_decay {weight_decay} \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints
"""

                jobs.append(job)
                print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
