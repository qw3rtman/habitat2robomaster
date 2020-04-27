import numpy as np
from pathlib import Path

jobs = list()

job = f"""python create_pointnav_dataset.py \\
    --scenes_dir /scratch/cluster/nimit/habitat/habitat-api/data/scene_datasets/gibson \\
    --split train
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
