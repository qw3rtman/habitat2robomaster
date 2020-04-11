import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', '-d', type=Path, required=True)
parsed = parser.parse_args()

lwns, lwns_norm, imgs = [], [], []
for d in parsed.dataset_dir.glob('00*'):
    csv_f = d / 'info.csv'
    if csv_f.exists():
        x = pd.read_csv(csv_f).iloc[0]
        lwns.append(x['lwns'])
        lwns_norm.append(x['lwns_norm'])
        imgs.append(len(list(d.iterdir())) - 2)

print(len(lwns))
print()

plt.hist(lwns); print(np.mean(lwns)), print(np.median(lwns))
print()

plt.hist(lwns_norm); print(np.mean(lwns_norm)), print(np.median(lwns_norm))
print()

print(np.sum(imgs), np.mean(imgs))
