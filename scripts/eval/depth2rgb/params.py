import numpy as np
from pathlib import Path

jobs = list()

"""
('/scratch/cluster/nimit/wandb/run-20200503_073134-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v5.03-aug/model_040.t7', True),
('/scratch/cluster/nimit/wandb/run-20200503_073133-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.005-0.0001-v5.03-aug/model_030.t7', True),
('/scratch/cluster/nimit/wandb/run-20200503_073133-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.005-0.0005-v5.03-aug/model_050.t7', True),
('/scratch/cluster/nimit/wandb/run-20200503_053929-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_230.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_053925-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_220.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_053924-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_190.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_053924-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_200.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_053921-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_170.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_053929-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_240.t7', False),
('/scratch/cluster/nimit/wandb/run-20200503_073134-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v5.03-aug/model_050.t7', True),
('/scratch/cluster/nimit/wandb/run-20200503_073731-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0001-v5.03-aug/model_030.t7', True),
('/scratch/cluster/nimit/wandb/run-20200503_073731-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0001-v5.03-aug/model_040.t7', True)
"""
models = [
    #('/scratch/cluster/nimit/wandb/run-20200503_184704-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0001-v5.03-aug/model_040.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200503_073134-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v5.03-aug/model_070.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_150.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200503_184704-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0001-v5.03-aug/model_050.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200503_073731-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0001-v5.03-aug/model_050.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_180.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200504_051554-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_010.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200503_073134-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v5.03-aug/model_080.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_024251-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_230.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200504_071432-resnet50-bc-backprop-pretrained-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_040.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_071401-se_resneXt50-bc-backprop-pretrained-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_040.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_071352-resnet50-bc-backprop-pretrained-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_010.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_075102-se_resneXt50-bc-backprop-pretrained-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_030.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_071324-resnet50-bc-backprop-pretrained-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_020.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_260.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200504_024251-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_270.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200503_073114-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.005-0.0005-v5.03-aug/model_100.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200503_184704-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0001-v5.03-aug/model_090.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200503_073133-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.005-0.0005-v5.03-aug/model_080.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_190.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_190.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_190.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_190.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-0.0005-v4.27/model_170.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200504_002639-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_200.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_023055-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_220.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200503_073134-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v5.03-aug/model_090.t7', True),
    #('/scratch/cluster/nimit/wandb/run-20200505_025804-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_100.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_025510-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_090.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_023055-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_230.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_240.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_240.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_400.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200503_073114-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.005-0.0005-v5.03-aug/model_120.t7', True)
    #('/scratch/cluster/nimit/wandb/run-20200505_170947-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-0.0005-v4.27/model_190.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170041-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.0001-5e-05-v4.27/model_310.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_320.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_340.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-0.0005-v4.27/model_120.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-0.0005-v4.27/model_250.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-0.0005-v4.27/model_140.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_260.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_171008-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_150.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-5e-05-v4.27/model_120.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-0.0005-v4.27/model_250.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_022400-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_270.t7', False),
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-resnet50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.001-0.0005-v4.27/model_130.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_250.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_260.t7', False)
    #('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_260.t7', False)
    ('/scratch/cluster/nimit/wandb/run-20200505_170505-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_270.t7', False)

    #('/scratch/cluster/nimit/wandb/run-20200503_054756-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-8-0.0001-5e-05-v4.27/model_150.t7', False),
]

for (model, compass) in models:
    job = f"""GLOG_minloglevel=2 MAGNUM_LOG=quiet python eval_hc.py \\
    --model {model} \\
    --split val \\
    {'--compass' if compass else ''}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
