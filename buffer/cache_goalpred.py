from pathlib import Path
from buffer.dataset import get_dataset

data_train, data_val = get_dataset(Path('/scratch/cluster/nimit/data/habitat/replica-apartment_0'), 'semantic', 'apartment_0', zoom=3, steps=8)
