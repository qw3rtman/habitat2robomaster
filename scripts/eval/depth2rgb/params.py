import numpy as np
from pathlib import Path

jobs = list()

models = [
    #'/scratch/cluster/nimit/wandb/run-20200502_031205-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_100.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_061959-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_060.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_061932-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_070.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_031205-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-32-0.0001-5e-05-v4.27/model_030.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_061932-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_070.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_061944-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-0.0005-v4.27/model_050.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_061959-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_060.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_092113-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_080.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_092113-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_100.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_090.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_150.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_110.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_140.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_130.t7'
    #'/scratch/cluster/nimit/wandb/run-20200502_173301-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_090.t7',
    #'/scratch/cluster/nimit/wandb/run-20200502_222057-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_160.t7',
    '/scratch/cluster/nimit/wandb/run-20200502_222028-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_100.t7'
]

for model in models:
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python eval_hc.py \\
    --model {model} \\
    --split val
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
