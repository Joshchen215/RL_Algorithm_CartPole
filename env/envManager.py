import gymnasium as gym
import torch


class CartPoleEnvManager:
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        self.env.reset()
        self.current_state = None
        self.done = False

    def reset(self):
        self.current_state = self.env.reset()[0]

    def take_action(self, action):
        self.current_state, reward, terminated, truncated, _ = self.env.step(action.item())
        self.done = terminated or truncated
        return torch.tensor([reward], device=self.device)

    def get_state(self):
        if self.done:
            return torch.zeros_like(torch.tensor(self.current_state, device=self.device)).float()
        else:
            return torch.tensor(self.current_state, device=self.device)
