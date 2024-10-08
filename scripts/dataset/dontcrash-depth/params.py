import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
    --task dontcrash \\
    --proxy depth \\
    --mode teacher \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth-nostop \\
    --num_episodes 1200
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
