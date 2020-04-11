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

"""
import plotly.graph_objects as go

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, 20)]

fig = go.Figure(data=[go.Box(y=lwns_norm,
    boxpoints='all', # can also be outliers, or suspectedoutliers, or False
    jitter=0.05, # add some jitter for a better separation between points
    pointpos=-1.6, # relative position of points wrt box
    name=f'{1}',
    boxmean=True,
    marker_color=c[17]
)])

fig.update_layout(
    xaxis=dict(title='Epoch', showgrid=False, zeroline=False, dtick=1),
    yaxis=dict(zeroline=False, gridcolor='white', range=[0., 1.]),
    paper_bgcolor='rgb(233,233,233)',
    plot_bgcolor='rgb(233,233,233)',
    showlegend=False
)

fig.show()
"""
