import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

job = f"""python buffer/generate_offline_samples.py \\
    --dataset_dir /scratch/cluster/nimit/data/habitat/replica-apartment_0 \\
    --mode greedy \\
    --dataset replica \\
    --split train \\
    --scene apartment_0 \\
    --num_episodes 10000 \\
    --target semantic \\
    --height 160 \\
    --width 384 \\
    --fov 120 \\
    --camera_height 0.25
"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
