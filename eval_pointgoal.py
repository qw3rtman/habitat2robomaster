from habitat_wrapper import Rollout, METRICS
from habitat_dataset import HabitatDataset
from model import get_model

from habitat_sim.utils.common import d3_40_colors_rgb

import argparse
from collections import defaultdict
from operator import itemgetter
import time

import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from pyquaternion import Quaternion
from PIL import Image, ImageDraw, ImageFont
import yaml

def get_model_args(model, key):
    return yaml.load((model.parent / 'config.yaml').read_text())[key]['value']

def get_env(model, proxy, scene, rnn=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student_args = get_model_args(model, 'student_args')
    data_args = get_model_args(model, 'data_args')
    net = get_model(**student_args, target=proxy, rnn=rnn, batch_size=data_args['batch_size']).to(device)
    print(device)
    net.load_state_dict(torch.load(model, map_location=device))

    teacher_args = get_model_args(model, 'teacher_args')
    teacher_args['proxy'] = proxy
    env = Rollout(**teacher_args, student=net, split='val', mode='student', rnn=rnn, shuffle=False, dataset=scene, sensors=['RGB_SENSOR', 'DEPTH_SENSOR'])
    env.mode = 'teacher'

    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', '-n', type=int, default=5)
    parser.add_argument('--models_root', '-r', type=Path, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--epoch', '-e', type=int)
    parser.add_argument('--proxy', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--rnn', action='store_true')
    parser.add_argument('--auto', '-a', action='store_true')
    parser.add_argument('--human', '-hr', action='store_true')
    parser.add_argument('--display', '-d', action='store_true')
    parsed = parser.parse_args()

    model_path = parsed.models_root / parsed.model
    if parsed.epoch is None:
        model = [model.stem for model in model_path.glob('model_*.t7')][-1] + '.t7'
        model_path = model_path / model
    else:
        model_path = model_path / f'model_{parsed.epoch:03}.t7'

    summary = defaultdict(list)
    dfgs = []

    task = get_model_args(model_path, 'teacher_args')['task']
    env = get_env(model_path, parsed.proxy, parsed.scene, rnn=parsed.rnn)
    for ep in range(parsed.num_episodes):
        print(f'[!] Start Episode {ep:06}')

        last_dfg = 0
        for i, step in enumerate(env.rollout()):
            source_position = env.state.position
            rot = env.state.rotation.components
            source_rotation = Quaternion(*rot[1:4], rot[0])
            goal_position = env.env.current_episode.goals[0].position
            direction = HabitatDataset.get_direction(source_position, source_rotation, goal_position).unsqueeze(dim=0)
            distance = np.linalg.norm(direction)
            print(f' [*] Step {i: >3}, Direction: ({direction[0,0].item(): >6.2f}, {direction[0,1].item(): >6.2f})   {distance: >5.2f}', end='')

            last_dfg = distance

            if step['is_stuck']:
                print(' (stuck?)')
                #break
            else:
                print()

            if parsed.display:# and (not parsed.auto or i % 5 == 0):
                frame = Image.fromarray(step['rgb'])
                draw = ImageDraw.Draw(frame)
                font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf', 20)
                draw.text((0, 0), '({: <5.1f}, {: <5.1f}) {: <4.1f}'.format(*direction[0], distance), (0, 0, 0), font=font)
                cv2.imshow('rgb', np.uint8(frame))

                cv2.imshow('depth', step['depth'])

                if step['semantic'] is not None:
                    semantic_img = Image.new("P", (step['semantic'].shape[1], step['semantic'].shape[0]))
                    semantic_img.putpalette(d3_40_colors_rgb.flatten())
                    semantic_img.putdata((step['semantic'].flatten() % 40).astype(np.uint8))
                    semantic_img = semantic_img.convert("RGBA")
                    cv2.imshow('semantic', np.uint8(semantic_img))
                if cv2.waitKey(1 if parsed.auto else 0) == ord('x'):
                    break

        dfgs.append(last_dfg)
        spl, soft_spl, success, dtg = itemgetter('spl', 'softspl', 'success', 'distance_to_goal')(env.env.get_metrics())
        for m, v in env.env.get_metrics().items():
            if m in METRICS:
                summary[m].append(v)
        print(f'[!] Finish Episode {ep:06}, DTG: {dtg}, Success: {success}, SPL: {spl}, Soft SPL: {soft_spl}, Direction: {direction}\n')

    print('Mean: {}'.format({k: np.mean(v) for k, v in summary.items() if k in METRICS}))
    print('Median: {}'.format({k: np.median(v) for k, v in summary.items() if k in METRICS}))
    print('DFG mean: {}, DFG median: {}'.format(np.mean(dfgs), np.median(dfgs)))
    lt5 = (np.array(dfgs) < 0.5).sum()
    lt10 = (np.array(dfgs) < 1.0).sum()
    print('<0.5: {}, <1.0: {}'.format(lt5, lt10))
