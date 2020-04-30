import numpy as np
from pathlib import Path

jobs = list()

scenes = [scene.stem.split('.')[0] for scene in Path('/scratch/cluster/nimit/habitat/habitat-api/data/datasets/pointnav/gibson/v1/val_ddppo/content').glob('*.gz')]
for scene in scenes:
    # train/test/val for MP3D, save RGB + semantic
    job = f"""python eval_hc.py \\
            --model /scratch/cluster/nimit/wandb/run-20200427_224755-se_resneXt50-bc-backprop-depth2rgb-gibson-noaug-original-1.0-16-0.001-5e-05-v4.27/model_040.t7 \\
            --scene {scene}
    """

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
