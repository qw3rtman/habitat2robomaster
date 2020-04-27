from pathlib import Path
import pandas as pd

with open('category_mapping.tsv', 'r') as csv_f:
    x = pd.read_csv(csv_f, sep='\t')

labels = {}
for i, row in x.iterrows():
    _class = row['nyu40id']
    if _class == _class:
        _class = int(_class)
    _label = row['nyu40class']

    labels[_class] = _label

keys = sorted(labels)
for key in keys:
    print(key, labels[key])
