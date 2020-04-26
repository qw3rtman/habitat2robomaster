import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for dataset_dir, scene in [('/scratch/cluster/nimit/data/habitat/pointgoal-depth-office-val', 'office')]: # ('/scratch/cluster/nimit/data/habitat/pointgoal-depth-castle-val', 'castle')
    for method, batch_sizes in [('backprop', [8, 16])]: #('feedforward', [64, 128])]: ('tbptt', [8, 16])]:
        for resnet_model in ['resnet50']: #, 'se_resneXt50']: # NOTE: se_resneXt50 used for their RGB models
            for batch_size in batch_sizes:
                for lr in [1e-3, 1e-4]:
                    for weight_decay in [5e-4, 5e-5]:
                        for aug in ['']: #, '--augmentation']:
                            job = f"""python train_pointgoal_student_rnn.py \\
                --description {unique}-overfit \\
                --max_epoch 150 \\
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
                --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
                {aug}
            """

                            jobs.append(job)
                            print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
