import numpy as np
from pathlib import Path
from datetime import datetime

jobs = list()
unique = datetime.now().strftime("%-m.%d")

job = "python buffer/cache_goalpred.py"

jobs.append(job)
print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
