from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

scene = 'gibson'#-mini'
for hidden_size in [64, 128, 256, 512]:
    for batch_size in [64, 128]:
        for resnet_model in ['resnet18']:
            for lr in [2e-4]:
                for weight_decay in [3.8e-7]:
                    job = f"""ulimit -n 4096; PYTHONHASHSEED=0 python -m again.train_jointly \\
    --description {unique}-rgb-jointly-{scene}-v3 \\
    --max_epoch 1000 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/{scene} \\
    --dataset_size 1.0 \\
    --resnet_model {resnet_model} \\
    --hidden_size {hidden_size} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    --scene_bias
    """

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
