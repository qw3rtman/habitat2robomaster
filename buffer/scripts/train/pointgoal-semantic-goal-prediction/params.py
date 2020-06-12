import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [64, 128]:
    for resnet_model in ['resnet34']:
        for temperature in [0.1, 1.0, 10.0]:
            for lr in [1e-4, 1e-3]:
                for weight_decay in [5e-5]:
                    job = f"""ulimit -n 16384; wandb on; python -m buffer.train_goal_prediction \\
    --description {unique}-v1 \\
    --max_epoch 100 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir "/scratch/cluster/nimit/data/habitat/replica-apartment_0-k=3" \\
    --resnet_model {resnet_model} \\
    --temperature {temperature} \\
    --target semantic \\
    --scene apartment_0 \\
    --zoom 5 \\
    --steps 8 \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
"""

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
