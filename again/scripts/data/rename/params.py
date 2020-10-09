import numpy as np
from pathlib import Path
import argparse

root = Path('/scratch/cluster/nimit/data/habitat')

jobs = list()
for scene_dir in root.glob('gibson-*'):
    job = f"""python -m again.rename \\
        --scene_dir {scene_dir}
    """

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
