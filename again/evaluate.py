import wandb
import argparse
from pathlib import Path

import torch
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .model import PointGoalPolicy, InverseDynamics, TemporalDistance, SceneLocalization, PointGoalPolicyAux
from .wrapper import Rollout
from .dataset import polar1, polar2, rff

ACTIONS = ['F', 'L', 'R']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--dataset') # override's config.yaml
    parser.add_argument('--scene')   # ^^^
    parser.add_argument('--split', required=True)
    parser.add_argument('--aux_model', type=Path)
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}-{parsed.split}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ normal
    net = PointGoalPolicy(**config['model_args']['value']).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.eval()
    """

    if parsed.aux_model is not None:
        aux_config = yaml.load((parsed.aux_model.parent / 'config.yaml').read_text())
        aux_net = TemporalDistance(**aux_config['model_args']['value']).to(device)
        aux_net.load_state_dict(torch.load(parsed.aux_model, map_location=device))

    #aux_net = InverseDynamics(**config['aux_model_args']['value']).to(device)
    #aux_net = TemporalDistance(**config['aux_model_args']['value']).to(device)
    aux_net = SceneLocalization(**config['aux_model_args']['value']).to(device)
    #aux_net.load_state_dict(torch.load(config['aux_model']['value'], map_location=device))

    net = PointGoalPolicyAux(aux_net, **config['model_args']['value']).to(device)
    net.load_state_dict(torch.load(parsed.model, map_location=device))
    net.eval()

    env = Rollout(shuffle=True, split='val', dataset='replica', scenes=parsed.scene)

    wandb.init(project='pointgoal-il-eval', name=run_name, config=config)
    wandb.run.summary['episode'] = 0

    n = 100
    success, spl, softspl = np.zeros(n), np.zeros(n), np.zeros(n)
    for ep in range(n):
        images = []

        for i, step in enumerate(env.rollout(net=net, goal_fn=polar1)):
            frame = Image.fromarray(step['rgb'])
            images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

        metrics = env.env.get_metrics()
        success[ep] = metrics['success']
        spl[ep] = metrics['spl']
        softspl[ep] = metrics['softspl']

        log = {
            f'{parsed.scene}_video': wandb.Video(np.array(images), fps=60, format='mp4'),
            'success_mean': success[:ep+1].mean(),
            'spl_mean': spl[:ep+1].mean(),
            'softspl_mean': softspl[:ep+1].mean()
        }

        wandb.run.summary['episode'] += 1
        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in log.items()},
                step=wandb.run.summary['episode'])
