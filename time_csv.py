import pandas as pd
import time

csv_f = '/u/nimit/Documents/robomaster/habitat2robomaster/test222/train/Maryhill-train_ddppo-000001/info.csv'
#csv_f = '/u/nimit/Documents/robomaster/habitat2robomaster/test222/train/Maryhill-train_ddppo-000001/episode.csv'

print('\npandas')
for _ in range(10):
    start=time.time()
    pd.read_csv(csv_f)
    print(time.time()-start)


print('\nraw')
for _ in range(10):
    start=time.time()
    with open(csv_f, 'r') as f:
        """
        lines = f.readlines()
        for line in lines:
            line.split(',')
        """
        print(f.readlines()[1].split(','))
    print(time.time()-start)
