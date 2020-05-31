import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for method, batch_sizes in [('feedforward', [32, 64])]:
    for resnet_model in ['se_resneXt101']:
        for batch_size in batch_sizes:
            for lr in [1e-3, 1e-4]:
                for weight_decay in [0.0]:
                    job = f"""python buffer/train.py \\
    --description {unique}-v6 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --resnet_model {resnet_model} \\
    --history_size 5 \\
    --hidden_size 1024 \\
    --method {method} \\
    --dataset castle \\
    --goal polar \\
    --proxy depth \\
    --target rgb \\
    --max_epoch 1000 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
"""

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
