import wandb
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
import yaml

from .model import PointGoalPolicy, InverseDynamics, TemporalDistance, PointGoalPolicyAux
from .buffer import ReplayBuffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--buffer', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parsed = parser.parse_args()

    config = yaml.load((parsed.model.parent / 'config.yaml').read_text())
    run_name = f"{config['run_name']['value']}-model_{parsed.epoch:03}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #aux_net = TemporalDistance(**config['aux_model_args']['value']).to(device)
    aux_net = InverseDynamics(**config['aux_model_args']['value']).to(device)
    #aux_net.load_state_dict(torch.load(config['aux_model']['value'], map_location=device))

    net = PointGoalPolicyAux(aux_net, **config['model_args']['value']).to(device)
    model = torch.load(parsed.model, map_location=device)
    net.load_state_dict(model)
    net.eval()

    # NOTE: we are finetuning this part
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(net.aux.parameters(), lr=2e-4, weight_decay=3.8e-7)

    with open(parsed.buffer, 'rb') as f:
        replay_buffer = pickle.load(f)

    wandb.init(project='pointgoal-il-finetune', name=run_name, config=config)
    wandb.run.summary['episode'] = 0

    net.aux.train()

    print(len(replay_buffer))

    for epoch in range(20):
        loss_mean = None
        correct, total = 0, 0
        for j, (t1, t2, action, distance) in enumerate(replay_buffer.get_dataset(iterations=200, batch_size=32, temporal_dim=1)): # NOTE: change based il or td
            print(f'train loop {j}')
            t1 = t1.to(device)
            t2 = t2.to(device)
            action = action.to(device)
            distance = distance.to(device)

            _distance = net.aux(t1, t2).logits
            #loss = criterion(_distance, distance)
            loss = criterion(_distance, action)
            loss_mean = loss.mean()

            #correct += (distance == _distance.argmax(dim=1)).sum().item()
            correct += (action == _distance.argmax(dim=1)).sum().item()
            total += t1.shape[0]

            loss_mean.backward()
            optim.step()
            optim.zero_grad()

        wandb.log({
            'accuracy': correct/total if total > 0 else 0,
            'loss': loss_mean.item()
        }, step=wandb.run.summary['episode'])
        wandb.run.summary['episode'] += 1

        torch.save(net.state_dict(), 'model_latest.t7')
