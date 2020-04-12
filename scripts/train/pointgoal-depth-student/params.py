import numpy as np
from pathlib import Path

jobs = list()

for resnet_model in ['resnet18', 'resnet50', 'se_resneXt50']:
    for dataset_size, interpolate_flag, interpolate_name in [(1.0, '', 'original')]:#, (0.3, '--interpolate', 'interpolate')]:
        for batch_size in [64, 128]: # 32
            for lr in [1e-3, 1e-4]:
                for weight_decay in [5e-5]: # 5e-4
                    #for dagger in ['', '--dagger'] if dataset_size == 1.0 else ['']:
                    for dagger in ['']:
                        for conditional_flag, conditional_name in [('--conditional', 'conditional')]:
                            for episodes_per_epoch in [100]:
                                for capacity in [1000]:
                                    job = f"""python train_pointgoal_student.py \\
    --resnet_model {resnet_model} \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth \\
    --dataset_size {dataset_size} \\
    --teacher_task pointgoal \\
    --teacher_proxy depth \\
    --dagger_dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-dagger/{resnet_model}-{conditional_name}-dagger-{interpolate_name}-{episodes_per_epoch}-{capacity}-{dataset_size}-{batch_size}-{lr}-{weight_decay} \\
    --lr {lr} \\
    --batch_size {batch_size} \\
    --weight_decay {weight_decay} \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    {dagger} {interpolate_flag} {conditional_flag}
"""

                                    jobs.append(job)
                                    print(job)

np.random.shuffle(jobs)
print(len(jobs))
JOBS = jobs[:len(jobs)]
