import torch
from pathlib import Path

from habitat_wrapper import Rollout, replay_episode
from buffer.frame_buffer import ReplayBuffer


env = Rollout('pointgoal', 'depth', 'rgb', mode='teacher',
        shuffle=False, split='train', dataset='gibson')

replay_buffer = ReplayBuffer(2**18, dshape=(256,256,3), dtype=torch.uint8)
for i in range(1328): # 332 * 4
    print(i)

    replay_episode(env, replay_buffer)
    if replay_buffer.size == replay_buffer.buffer_size:
        print(f'done, early! at {i}')
        break

root = Path('/scratch/cluster/nimit/data/habitat/depth2rgb_buffer')
replay_buffer.save(root)
