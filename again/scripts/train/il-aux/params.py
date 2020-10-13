from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

models = [
    (512, Path('/scratch/cluster/nimit/models/sl/2x2/512/model_best.t7')),
    (256, Path('/scratch/cluster/nimit/models/sl/2x2/256/model_best.t7')),
    (64, Path('/scratch/cluster/nimit/models/sl/2x2/64/model_best.t7')),
]

scene = 'gibson'#-mini'
for hidden_size, model in models:
    for batch_size in [16, 32, 64]:
        for resnet_model in ['resnet18']:
            for lr in [2e-4, 2e-3]:
                for weight_decay in [3.8e-7]:
                    job = f"""ulimit -n 4096; PYTHONHASHSEED=0 python -m again.train_il \\
    --description {unique}-rgb-single_encoder-sl-aux-{scene}-v15 \\
    --max_epoch 2000 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/{scene} \\
    --dataset_size 1.0 \\
    --aux_model {model} \\
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
