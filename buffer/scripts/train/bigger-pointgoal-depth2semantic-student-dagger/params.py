import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for supervision in ['greedy', 'ddppo']:
    for method, batch_sizes in [('feedforward', [64, 128])]:
        for resnet_model in ['se_resneXt50']:
            for batch_size in batch_sizes:
                for lr in [1e-3, 1e-4]:
                    for weight_decay in [0.0]:
                        job = f"""python buffer/train.py \\
    --description "k=3-{unique}-v1" \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --hidden_size 512 \\
    --resnet_model {resnet_model} \\
    --history_size 1 \\
    --supervision {supervision} \\
    --method {method} \\
    --dagger 3 \\
    --dataset replica \\
    --scene apartment_0 \\
    --goal polar \\
    --proxy depth \\
    --target semantic \\
    --max_epoch 1000 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    --height 160 \\
    --width 384 \\
    --fov 120 \\
    --camera_height 0.25
"""

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
