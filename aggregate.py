from pathlib import Path
import pandas as pd

csv_fs = list(Path('eval_results').glob('*.csv'))

best = 0
best_scene = ''

total = 0
for csv_f in csv_fs:
    with open(csv_f, 'r') as f:
        summary = pd.read_csv(f)

    total += (summary / 10)

    softspl = summary['softspl'].iloc[0]/10
    if softspl > best:
        best = softspl
        best_scene = csv_f.stem

print(total / len(csv_fs))
print(best, best_scene)
