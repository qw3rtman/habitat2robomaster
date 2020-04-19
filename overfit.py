from habitat_dataset import get_dataset
import torch
from model import ConditionalStateEncoderImitation

batch_size = 2
net = ConditionalStateEncoderImitation(batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

optim = torch.optim.Adam(net.parameters())
criterion = torch.nn.CrossEntropyLoss()
net.train()

print('train...')
data_train, data_val = get_dataset('test', rnn=True, batch_size=batch_size)
for x in data_train:
    net.clean()
    optim.zero_grad()

    rgb, action, prev_action, meta, mask = x
    episode_loss = torch.zeros(rgb.shape[0]).to(device)
    for t in range(rgb.shape[0]):
        _action = net((rgb[t], meta[t], prev_action[t], mask[t]))

        loss = criterion(_action, action[t])
        episode_loss[t] = loss

    loss_mean = episode_loss.mean()
    print(loss_mean)
    loss_mean.backward()
    optim.step()

print('\n\nval...')
net.eval()
net.batch_size = 1

data_train, data_val = get_dataset('test', rnn=True, batch_size=1)
for x in data_train:
    net.clean()

    rgb, action, prev_action, meta, mask = x
    episode_loss = torch.zeros(rgb.shape[0]).to(device)
    for t in range(rgb.shape[0]):
        _action = net((rgb[t], meta[t], prev_action[t], mask[t]))

        loss = criterion(_action, action[t])
        episode_loss[t] = loss

    loss_mean = episode_loss.mean()
    print(loss_mean)
