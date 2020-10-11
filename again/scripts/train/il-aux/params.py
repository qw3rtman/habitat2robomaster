from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

scene = 'gibson'#-mini'
for batch_size in [16, 32]:
    for hidden_size in [512]: # NOTE: pick based on model!
        for resnet_model in ['resnet18']:
            for lr in [2e-4, 2e-5]:
                for weight_decay in [3.8e-7]:
                    job = f"""ulimit -n 4096; PYTHONHASHSEED=0 python -m again.train_il \\
    --description {unique}-rgb-single_encoder-sl-aux-{scene}-v8 \\
    --max_epoch 2000 \\
    --checkpoint_dir /scratch/cluster/nimit/checkpoints \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/{scene} \\
    --dataset_size 1.0 \\
    --aux_model /scratch/cluster/nimit/models/sl/bias_xyt/512/model_166.t7 \\
    --resnet_model {resnet_model} \\
    --hidden_size {hidden_size} \\
    --batch_size {batch_size} \\
    --lr {lr} \\
    --weight_decay {weight_decay} \\
    #--aux_model /scratch/cluster/nimit/models/sl/nobias_xyt/256/model_300.t7 \\
    """

                    jobs.append(job)
                    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
