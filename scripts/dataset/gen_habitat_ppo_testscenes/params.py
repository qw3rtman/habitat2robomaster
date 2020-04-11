import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
    --mode teacher \\
    --proxy depth \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/dontcrash-depth \\
    --num_episodes 1200
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
