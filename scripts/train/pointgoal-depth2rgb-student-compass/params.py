import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for dataset_dir, scene in [('/scratch/cluster/nimit/data/habitat/pointgoal-depth2rgb', 'gibson')]: # validate in gibson habitat challenge 2019 val
    for method, batch_sizes in [('backprop', [8, 16])]: #, ('tbptt', [8, 16])]: # ('feedforward', [64, 128])
        for resnet_model in ['se_resneXt50']: # NOTE: se_resneXt50 used for their RGB models
            for batch_size in batch_sizes:
                for lr in [1e-3, 5e-3]:
                    for weight_decay in [1e-4, 5e-4]:
                        job = f"""python train_pointgoal_student_rnn.py \\
    --description {unique}-aug \\
    --max_epoch 400 \\
    --resnet_model {resnet_model} \\
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
