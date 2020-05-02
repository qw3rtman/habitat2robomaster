from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', type=Path, nargs='+', required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--destination', '-d', type=Path, required=True)
parsed = parser.parse_args()

ep = 1
for source in parsed.source:
    for scene in source.glob(f'*{parsed.split}'):
        episodes = list(scene.iterdir())
        for i, episode in enumerate(episodes):
            if episode.is_dir():
                print(episode)
                if i >= len(episodes) - 1: # last episode
                    split = 'val'
                else:
                    split = 'train'

                episode_name = episode.stem
                print(episode_name)

                #(parsed.destination / split / f'{ep:06}').symlink_to(episode)
                (parsed.destination / split / f'{episode_name}-semantic').symlink_to(episode)
                ep += 1
