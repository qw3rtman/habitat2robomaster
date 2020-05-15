import argparse
from pathlib import Path
import shutil
import wandb
import sys

from train_pointgoal_student_rnn_dagger import get_run_name
from habitat_wrapper import TASKS, MODELS, MODALITIES

root = Path('/scratch/cluster/nimit')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=Path, default='checkpoints')
    parser.add_argument('--teacher_task', choices=TASKS, required=True)
    parser.add_argument('--proxy', choices=MODELS.keys(), required=True)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--target', choices=MODALITIES, required=True)
    parser.add_argument('--resnet_model', choices=['resnet18', 'resnet50', 'resneXt50', 'se_resnet50', 'se_resneXt101', 'se_resneXt50'])
    parser.add_argument('--method', type=str, choices=['feedforward', 'backprop', 'tbptt'], default='backprop', required=True)
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--dagger_dir', type=Path, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--dataset_size', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parsed = parser.parse_args()
    run_name = get_run_name(parsed)
    train_args = ' '.join(sys.argv[1:])

    api = wandb.Api()

    print(run_name, parsed.dagger_dir)
    checkpoint = root/'checkpoints'/run_name
    while True:
        # 1. run train script
        print('training...')
        subprocess.run(['python', 'train_pointgoal_dagger_student_rnn.py']+sys.argv[1:])

        # 2. copy config.yaml to checkpoint dir
        run_dirs = list((root/'wandb').glob(f'*{run_name}'))
        run_dirs.sort(key=os.path.getmtime)
        shutil.copy(run_dirs[-1]/'config.yaml', checkpoint/'config.yaml')

        # 3. get epoch
        run = api.run('qw3rtman/habitat-pointgoal-depth-student/{run_name}')

        # 4. run collect_dagger script
        print('dagger...')
        subprocess.run(['python', 'dagger_collector.py', '--model', checkpoint/'model_latest.t7', '--epoch', run.summary['epoch']])
