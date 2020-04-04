import numpy as np
from pathlib import Path

jobs = list()

for input_type, num_episodes in [('rgb', 1500), ('depth', 2000)]:
    job = f"""python habitat_wrapper.py \\
    --input_type {input_type} \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/ppo_{input_type}  \\
    --num_episodes {num_episodes}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
