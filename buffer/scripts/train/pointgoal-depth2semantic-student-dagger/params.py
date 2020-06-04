import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for aug in ['--augmentation']:
    for method, batch_sizes in [('feedforward', [64, 128])]:
        for resnet_model in ['se_resneXt50']:
            for batch_size in batch_sizes:
                for lr in [1e-3, 1e-4]:
                    for weight_decay in [0.0]:
                        job = f"""python buffer/train.py \\
    --description {unique}-v3 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --hidden_size 128 \\
    --resnet_model {resnet_model} \\
    --history_size 1 \\
    --method {method} \\
    --dagger \\
    --dataset office \\
    --goal polar \\
    --proxy depth \\
    --target semantic \\
    --max_epoch 1000 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    {aug}
"""

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
