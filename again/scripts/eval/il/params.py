import numpy as np
from pathlib import Path
import argparse

gibson = {
    'train': ['Adrian', 'Albertville', 'Anaheim', 'Andover', 'Angiola', 'Annawan', 'Applewold', 'Arkansaw', 'Avonia', 'Azusa', 'Ballou', 'Beach', 'Bolton', 'Bowlus', 'Brevort', 'Capistrano', 'Colebrook', 'Convoy', 'Cooperstown', 'Crandon', 'Delton', 'Dryville', 'Dunmor', 'Eagerville', 'Goffs', 'Hainesburg', 'Hambleton', 'Haxtun', 'Hillsdale', 'Hometown', 'Hominy', 'Kerrtown', 'Maryhill', 'Mesic', 'Micanopy', 'Mifflintown', 'Mobridge', 'Monson', 'Mosinee', 'Nemacolin', 'Nicut', 'Nimmons', 'Nuevo', 'Oyens', 'Parole', 'Pettigrew', 'Placida', 'Pleasant', 'Quantico', 'Rancocas', 'Reyno', 'Roane', 'Roeville', 'Rosser', 'Roxboro', 'Sanctuary', 'Sasakwa', 'Sawpit', 'Seward', 'Shelbiana', 'Silas', 'Sodaville', 'Soldier', 'Spencerville', 'Spotswood', 'Springhill', 'Stanleyville', 'Stilwell', 'Stokes', 'Sumas', 'Superior', 'Woonsocket'],
    'val': ['Denmark', 'Greigsville', 'Sands', 'Eastville', 'Ribera', 'Edgemere', 'Scioto', 'Swormville', 'Pablo', 'Sisters', 'Mosquito', 'Cantwell', 'Eudora', 'Elmira']
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--split', choices=['train', 'val'], required=True)
args = parser.parse_args()

jobs = list()
for scene in gibson[args.split]:
    job = f"""python -m again.evaluate \
--model {args.model} \\
--epoch {args.epoch} \\
--dataset gibson \\
--scene {scene} \\
--split {args.split}
"""

    jobs.append(job)
    print(job)

print(len(jobs))
JOBS = jobs[:len(jobs)]
