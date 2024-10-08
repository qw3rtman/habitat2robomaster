import argparse
import collections

import torch

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

ckpt = torch.load(args.path, map_location="cpu")
state = ckpt["state_dict"]

baseline_weight_mapping = {
    "actor_critic.net.rnn": "actor_critic.net.state_encoder.rnn",
    "actor_critic.net.critic_linear": "actor_critic.critic.fc",
    "actor_critic.net.cnn": "actor_critic.net.visual_encoder.cnn",
}


def _map_weights(name):
    for k, v in baseline_weight_mapping.items():
        if name.startswith(k):
            return name.replace(k, v)

    return name


new_state = collections.OrderedDict()
for k, v in state.items():
    new_state[_map_weights(k)] = v

state = new_state

reordering = torch.tensor([3, 0, 1, 2], dtype=torch.long)
for k in [
    "actor_critic.action_distribution.linear.weight",
    "actor_critic.action_distribution.linear.bias",
]:
    state[k] = state[k][reordering]


ckpt["state_dict"] = state
torch.save(ckpt, args.output if args.output is not None else args.path)
