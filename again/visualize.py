import wandb

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .wrapper import Rollout

BACKGROUND = (0,0,0,0)
COLORS = [
    (0,47,0,150),
    (253,253,17,150)
]

ACTIONS = ['S', 'F', 'L', 'R']

if __name__ == '__main__':
    wandb.init(project='test')

    env = Rollout(shuffle=True, split='train', dataset='gibson')

    images = []
    for i, step in enumerate(env.rollout()):
        frame = Image.fromarray(step['rgb'])

        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 18)
        draw.rectangle((0, 0, 255, 20), fill='black')

        _action = ACTIONS[step['action']['action']]
        draw.text((0, 0), '{: <4.1f} {}'.format(env.env.get_metrics()['distance_to_goal'], _action), fill='white', font=font)

        images.append(np.transpose(np.uint8(frame), (2, 0, 1)))

    wandb.log({
        f'video': wandb.Video(np.array(images), fps=20, format='mp4')
    })
