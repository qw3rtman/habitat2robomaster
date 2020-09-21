import wandb
import argparse
from pathlib import Path

import torch
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from .model import PointGoalPolicy
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
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}-{parsed.split}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = PointGoalPolicy(**config['model_args']['value'], **config['data_args']['value']).to(device)
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
            f'{parsed.scene}_video': wandb.Video(np.array(images), fps=20, format='mp4'),
            'success_mean': success[:ep+1].mean(),
            'spl_mean': spl[:ep+1].mean(),
            'softspl_mean': softspl[:ep+1].mean()
        }

        wandb.run.summary['episode'] += 1
        wandb.log(
                {('%s/%s' % ('val', k)): v for k, v in log.items()},
                step=wandb.run.summary['episode'])
