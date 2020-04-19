import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
    --task pointgoal \\
    --proxy depth \\
    --mode teacher \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-office \\
    --num_episodes 1000
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
