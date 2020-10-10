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
parser.add_argument('--model_dir', type=Path, required=True)
args = parser.parse_args()

config = yaml.load((args.model_dir / 'config.yaml').read_text())
net = again.model.SceneLocalization(**config['model_args']['value']) # no need to put on GPU

(Path('svd')/args.model_dir.stem).mkdir(parents=True, exist_ok=True)
for model in args.model_dir.glob('*.t7'):
    net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))

    w = torch.stack([scene_fc.weight for scene_fc in net.scene_fc.values()], axis=0).cpu().detach().numpy()
    S = np.array([np.linalg.svd(w_)[1] for w_ in w])

    plt.scatter(*S.T)
    for idx, name in enumerate(GIBSON_IDX2NAME):
        plt.annotate(name, (S[idx][0], S[idx][1]))

    #m, b = np.polyfit(S[:,0], S[:,1], 1)
    m, b, r, p, stderr = linregress(S[:,0], S[:,1])
    x = np.linspace(S[:,0].min(), S[:,0].max(), 100)
    plt.plot(x, (m*x)+b, c='r')
    plt.title(f'R = {r:.02f}, p = {p:.02e}, y = {m:.03f}x+{b:.03f}, stderr = {stderr:.03f}')
    print(r, p, stderr)

    plt.xlim(0.25, 1.75)
    plt.ylim(0.25, 1.25)

    #plt.show()
    epoch = model.stem.split('_')[1].split('.')[0]
    plt.savefig(Path('svd')/args.model_dir.stem/f'{epoch}.png')
    plt.clf()
