import torch
from typing import NamedTuple
from agent_dqn.conf.conf import Config


class Processor:
    @staticmethod
    def convert_tensors(experiences: NamedTuple):
        batch = Config.Experience(*zip(*experiences))
        states = torch.stack(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.stack(batch.next_state)
        rewards = torch.cat(batch.reward)
        dones = torch.cat(batch.done)
        return (states, actions, next_states, rewards, dones)