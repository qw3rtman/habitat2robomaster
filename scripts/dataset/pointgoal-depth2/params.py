import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
    --task pointgoal \\
    --proxy depth \\
    --mode teacher \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-office-val \\
    --num_episodes 100
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
