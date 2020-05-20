import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for dataset_dir, scene in [('/scratch/cluster/nimit/data/habitat/pointgoal-depth2rgb', 'gibson')]: # validate in gibson habitat challenge 2019 val
    for method, batch_sizes in [('feedforward', [64, 128])]:
        for resnet_model in ['se_resneXt50']: # NOTE: se_resneXt50 used for their RGB models
            for batch_size in batch_sizes:
                for lr in [1e-3, 1e-4]:
                    for weight_decay in [0.0, 5e-5]:
                        job = f"""python dagger_controller.py \\
    --description {unique}-6 \\
    --max_epoch 1000 \\
    --resnet_model {resnet_model} \\
    --augmentation \\
    --dataset_dir {dataset_dir} \\
    --scene {scene} \\
    --method {method} \\
    --teacher_task pointgoal \\
    --proxy depth \\
    --target rgb \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --weight_decay {weight_decay} \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints
"""

                        jobs.append(job)
                        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
