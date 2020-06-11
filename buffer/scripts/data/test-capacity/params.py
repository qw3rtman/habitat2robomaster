import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

job = "python -c 'import torch; torch.ones((780000, 160, 384), dtype=torch.uint8)'"

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
