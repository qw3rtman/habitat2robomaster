import time
import habitat_dataset
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, required=True)
parser.add_argument('--rnn', action='store_true')
parser.add_argument('--semantic', action='store_true')
parsed = parser.parse_args()

root = f'/scratch/cluster/nimit/data/habitat/pointgoal-depth2{"semantic" if parsed.semantic else "rgb"}'
data_train, data_val = habitat_dataset.get_dataset(root, rnn=parsed.rnn, rgb=(not parsed.semantic), semantic=parsed.semantic, batch_size=parsed.batch_size)

total_frames = 0

start = time.time()
for i, x in enumerate(data_train):
    idx = 0
    if parsed.semantic:
        if parsed.rnn:
            idx = 1
        else:
            idx = 2

    total_frames += np.prod(x[idx].shape[:-3])
    print(x[idx].shape)
    if i == 3:
        break

elapsed = time.time()-start

print(elapsed, total_frames)
print((time.time()-start) / total_frames)
