from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

for batch_size in [16, 32]:
    for hidden_size in [512]: # NOTE: pick based on model
        for resnet_model in ['resnet18']:
            for lr in [2e-4, 2e-5]:
                for weight_decay in [3.8e-7]:
                    job = f"""ulimit -n 4096; PYTHONHASHSEED=0 python -m again.train_il \\
    --description {unique}-rgb-single_encoder-sl-aux-v2SMALL \\
    --max_epoch 2000 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_0 \\
    --dataset_size 0.1 \\
    --aux_model /scratch/cluster/nimit/wandb/run-20201008_041056-3450328272347932749/model_010.t7 \\
    --resnet_model {resnet_model} \\
    --hidden_size {hidden_size} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay}
    """

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
