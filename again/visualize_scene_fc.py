import yaml
import again.model
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
from .const import GIBSON_IDX2NAME
from scipy.stats import linregress

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
args = parser.parse_args()

config = yaml.load((args.model.parent / 'config.yaml').read_text())
net = again.model.SceneLocalization(**config['model_args']['value']) # no need to put on GPU
net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

xy_fc = [fc.weight.cpu().detach().numpy() for scene, fc in net.scene_xy_fc.items()]
t_fc  = [fc.weight.cpu().detach().numpy() for scene, fc in net.scene_t_fc.items()]

plt.subplots_adjust(right=0.9, top=0.9, wspace=0.9, hspace=0.9)
fig, ax = plt.subplots(nrows=9, ncols=8, figsize=(10, 10))
for i in range(len(xy_fc)):
    print(np.linalg.matrix_rank(xy_fc[i]))
    print(xy_fc[i])
    print(xy_fc[i].T @ xy_fc[i])
    print()
    print(np.linalg.matrix_rank(t_fc[i]))
    print(t_fc[i])
    print(t_fc[i].T @ t_fc[i])
    print()
    print()
    print()
    print()
    row = i//8
    col = i - (row*8)
    ax[row][col].imshow(xy_fc[i], vmin=-1, vmax=1)
    ax[row][col].set_title(GIBSON_IDX2NAME[i])
    ax[row][col].axis('off')

plt.savefig(Path('2x2')/f'{args.model.parent.stem}-{args.model.stem}.png')
