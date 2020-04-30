import numpy as np
from pathlib import Path

jobs = list()

for target in ['rgb', 'semantic']:
    job = f"""python preprocess_episodes.py \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth2{target} \\
    --target {target}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
