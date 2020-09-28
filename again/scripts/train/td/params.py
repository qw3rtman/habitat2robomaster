import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for temporal_dim in [4]:#, 6]:
    for batch_size in [16, 32]:
        for hidden_size in [128, 256]:
            for resnet_model in ['resnet18']:
                for lr in [2e-4]:
                    for weight_decay in [3.8e-7]:
                        job = f"""ulimit -n 4096; python -m again.train_td \\
    --description {unique}-semantic-aux-v6 \\
    --max_epoch 500 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-frl_apartment_4 \\
    --resnet_model {resnet_model} \\
    --hidden_size {hidden_size} \\
    --temporal_dim {temporal_dim} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
    """

                        jobs.append(job)
                        print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
