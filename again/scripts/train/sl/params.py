import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for scene_bias in ['--scene_bias', '']:
    for dataset_size in [1.0]:
        for batch_size in [128, 256]: # can use big batch sizes here, not training a policy
            for hidden_size in [64]:
                for resnet_model in ['resnet18']:
                    for lr in [2e-4]:
                        for weight_decay in [3.8e-7]:
                            job = f"""ulimit -n 4096; PYTHONHASHSEED=0 python -m again.train_sl \\
        --description {unique}-gibson-rgb-single_encoder-v11 \\
        --max_epoch 2000 \\
        --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
        --dataset_dir /scratch/cluster/nimit/data/habitat/gibson \\
        --dataset_size {dataset_size} \\
        --batch_size {batch_size} \\
        --resnet_model {resnet_model} \\
        --hidden_size {hidden_size} \\
        --lr {lr} \\
        --weight_decay {weight_decay} \\
        {scene_bias}
        """

                            jobs.append(job)
                            print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
