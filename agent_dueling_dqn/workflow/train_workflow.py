import torch
from tqdm import tqdm
from itertools import count
from datetime import datetime
from agent_dueling_dqn.agent import Agent
from agent_dueling_dqn.conf.conf import Config
from env import envManager


def train_workflow(agent: Agent, env: envManager, num_episodes=Config.NUM_EPISODES):
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        env.reset()
        state = env.get_state()
        for duration in count():
            action = agent.predict(state)
            reward = env.take_action(action)
            next_state = env.get_state()
            env.done = torch.tensor([env.done or duration > 500]).float().to(agent.device)
            agent.algorithm.memory_push(Config.Experience(state, action, next_state, reward, env.done))
            state = next_state
            total_reward += reward
            if env.done or duration > 500:
                agent.monitor.add_duration_info(duration)
                agent.monitor.add_reward_info(total_reward)
                break

        if agent.algorithm.can_sample():
            experiences = agent.algorithm.memory_sample()
            agent.learn(experiences)

def test_episodes(num_episodes: int, agent: Agent, env: envManager):
    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        env.reset()
        state = env.get_state()
        for duration in count():
            action = agent.predict(state, exploit_flag=True)
            reward = env.take_action(action)
            total_reward += reward
            next_state = env.get_state()
            env.done = torch.tensor([env.done or duration > 500]).float().to(agent.device)
            state = next_state
            if env.done or duration > 500:
                agent.monitor.add_duration_info(duration)
                agent.monitor.add_reward_info(total_reward)
                break
