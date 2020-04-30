from habitat_dataset import get_dataset
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', '-d', type=Path, required=True)
parser.add_argument('--target', choices=['rgb', 'semantic'], required=True)
parsed = parser.parse_args()

# cache the list of rgb/semantic files, positions, meta, etc
get_dataset(parsed.dataset_dir, rnn=True, rgb=parsed.target=='rgb', semantic=parsed.target=='semantic', num_workers=0)
