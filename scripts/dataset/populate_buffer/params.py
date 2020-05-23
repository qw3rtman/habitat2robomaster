import numpy as np
from pathlib import Path

jobs = list()

job = f"""python populate_buffer.py"""

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
