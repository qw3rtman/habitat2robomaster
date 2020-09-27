import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [16, 32]:
    for hidden_size in [256]:#, 256]:
        for resnet_model in ['resnet18']:
            for lr in [2e-4]:
                for weight_decay in [3.8e-7]:
                    job = f"""ulimit -n 4096; python -m again.train_id \\
    --description {unique}-frl_apartment_4-semantic-single_encoder-v2 \\
    --max_epoch 200 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-frl_apartment_4 \\
    --resnet_model {resnet_model} \\
    --hidden_size {hidden_size} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    --aux_model /scratch/cluster/nimit/wandb/run-20200922_013536-9138119813105194381/files/model_200.t7 \\
    """

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
