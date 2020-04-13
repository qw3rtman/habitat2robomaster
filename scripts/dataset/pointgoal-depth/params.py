import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
    --task pointgoal \\
    --proxy depth \\
    --mode teacher \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/pointgoal-depth-new \\
    --num_episodes 1200
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
