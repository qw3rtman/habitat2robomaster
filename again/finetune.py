import wandb
import argparse
from pathlib import Path

import torch
import numpy as np
import yaml
import tqdm

from .model import SceneLocalization, PointGoalPolicyAux
from .pointgoal_dataset import get_dataset

def resume_project(net, optim, config):
    print('Resumed at epoch %d.' % wandb.run.summary['epoch'])

    net.load_state_dict(torch.load(Path(config['checkpoint_dir']['value']) / 'model_latest.t7'))

def checkpoint_project(net, optim, config):
    torch.save(net.state_dict(), Path(config['checkpoint_dir']['value']) / 'model_latest.t7')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aux_net = SceneLocalization(**config['aux_model_args']['value']).to(device)
    net = PointGoalPolicyAux(aux_net, **config['model_args']['value']).to(device)
    model = torch.load(parsed.model, map_location=device)
    net.load_state_dict(model)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(net.aux.parameters(), lr=2e-4, weight_decay=3.8e-7)

    wandb.init(project='pointgoal-il-finetune', name=run_name, config=config,
            resume=True, id=str(hash(run_name)))
    wandb.save(str(Path(wandb.run.dir) / '*.t7'))
    if wandb.run.resumed:
        resume_project(net, optim, config)
    else:
        wandb.run.summary['epoch'] = 0

        wandb.run.summary['best_accuracy'] = 0
        wandb.run.summary['best_epoch'] = 0

    data_train, data_val = get_dataset(num_workers=8, dataset_dir=parsed.dataset_dir, \
            batch_size=32, goal_fn='polar1')

    net.aux.train()

    for epoch in range(50):
        losses = list()

        correct, total = 0, 0
        iterator = tqdm.tqdm(data_train, total=len(data_train), position=1, leave=None)
        for j, (_, rgb, goal, action) in enumerate(iterator):
            rgb = rgb.to(device)
            goal = goal.to(device)
            action = action.to(device)

            _action = net(rgb, goal).logits

            loss = criterion(_action, action)
            loss_mean = loss.mean()

            correct += (action == _action.argmax(dim=1)).sum().item()
            total += rgb.shape[0]

            loss_mean.backward()
            optim.step()
            optim.zero_grad()

            losses.append(loss_mean.item())

        accuracy = correct/total if total > 0 else 0
        wandb.log({
            'accuracy': accuracy,
            'loss': np.mean(losses)
        }, step=wandb.run.summary['epoch'])
        wandb.run.summary['epoch'] += 1

        torch.save(net.state_dict(), 'model_latest.t7')
        if accuracy > wandb.run.summary['best_accuracy']:
            wandb.run.summary['best_accuracy'] = accuracy
            wandb.run.summary['best_epoch'] = epoch
            torch.save(net.state_dict(), Path(wandb.run.dir) / 'model_best.t7')

        checkpoint_project(net, optim, config)
        if epoch % 1 == 0:
            torch.save(net.state_dict(), Path(wandb.run.dir) / ('model_%03d.t7' % epoch))
