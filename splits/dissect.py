from pathlib import Path
import pandas as pd

with open('gibson_quality_ratings.csv', 'r') as csv_f:
    quality = pd.read_csv(csv_f)

with open('gibson_splits/train_val_test_fullplus.csv', 'r') as csv_f:
    splits = pd.read_csv(csv_f)

two_plus = quality.loc[quality['quality'] >= 2]
two_plus_splits = splits.loc[splits['id'].isin(set(two_plus.scene_id))]
print(two_plus_splits.loc[two_plus_splits['train'] == 1]['id'].tolist())
