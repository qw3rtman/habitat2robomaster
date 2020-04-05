import numpy as np
from pathlib import Path

jobs = list()

for resnet_model in ['resnet34']: # resnet18
    for lr in [1e-3, 1e-4]:
        for batch_size in [64, 128]: # 32
            for weight_decay in [5e-5]: # 5e-4
                job = f"""python train_image.py \\
    --network ddppo-direct \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \\
    --resnet_model {resnet_model} \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --weight_decay {weight_decay}
"""

                jobs.append(job)
                print(job)

np.random.shuffle(jobs)
print(len(jobs))
JOBS = jobs[:len(jobs)]
