import numpy as np
from pathlib import Path

jobs = list()

i = 1
for resnet_model in ['SE-ResNeXt-50']:
    for lr in [1e-3, 1e-4]:
        for batch_size in [64, 128]: # 32
            for weight_decay in [5e-5]: # 5e-4
                for d, per_epoch in [('habitat', 1000), ('habitat2', 5000)]:
                    job = f"""python train_image_dagger.py \\
        --network ddppo-direct \\
        --input_type depth \\
        --per_epoch {per_epoch} \\
        --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
        --dataset_dir /scratch/cluster/nimit/data/{d}/ddppo-direct-depth-dagger-{i} \\
        --lr {lr} \\
        --batch_size {batch_size} \\
        --weight_decay {weight_decay}
    """

                    jobs.append(job)
                    print(job)

                i += 1

np.random.shuffle(jobs)
print(len(jobs))
JOBS = jobs[:len(jobs)]
