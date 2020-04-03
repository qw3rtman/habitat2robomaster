import numpy as np
from pathlib import Path

jobs = list()

job = f"""python habitat_wrapper.py \\
--input_type depth \\
--dataset_dir /scratch/cluster/nimit/data/habitat/ppo_depth \\
--num_episodes 1000
"""

jobs.append(job)
print(job)

np.random.shuffle(jobs)
print(len(jobs))
JOBS = jobs[:len(jobs)]
