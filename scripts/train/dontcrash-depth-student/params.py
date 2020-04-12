import numpy as np
from pathlib import Path

jobs = list()

for resnet_model in ['resnet18', 'resnet50', 'se_resneXt50']:
    for dataset_size in [1.0]: # 0.1, 0.5
        for batch_size in [64, 128]: # 32
            for lr in [1e-3, 1e-4]:
                for weight_decay in [5e-5]: # 5e-4
                    #for dagger in ['', '--dagger'] if dataset_size == 1.0 else ['']:
                    for dagger in ['']:
                        for conditional_flag, conditional_name in [('', 'direct')]:#'--conditional']
                            for episodes_per_epoch in [100]:
                                for capacity in [1000]:
                                    job = f"""python train_image_dagger.py \\
    --resnet_model {resnet_model} \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth-nostop \\
    --dataset_size {dataset_size} \\
    --teacher_task dontcrash \\
    --teacher_proxy depth \\
    --dagger_dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth-dagger/{resnet_model}-{conditional_name}-dagger-{episodes_per_epoch}-{capacity}-{dataset_size}-{batch_size}-{lr}-{weight_decay} \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --weight_decay {weight_decay} \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    {dagger} {conditional_flag}
"""

                                    jobs.append(job)
                                    print(job)

np.random.shuffle(jobs)
print(len(jobs))
JOBS = jobs[:len(jobs)]
