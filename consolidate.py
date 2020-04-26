from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', type=Path, required=True)
parser.add_argument('--split', required=True)
parser.add_argument('--destination', '-d', type=Path, required=True)
parsed = parser.parse_args()

for scene in parsed.source.glob(f'*{parsed.split}'):
    episodes = list(scene.iterdir())
    for i, episode in enumerate(episodes):
        if episode.is_dir():
            if i >= len(episodes) - 3:
                split = 'val'
            else:
                split = 'train'

            episode_name = episode.stem
            print(episode_name)

            (parsed.destination / split / episode_name).symlink_to(episode)
