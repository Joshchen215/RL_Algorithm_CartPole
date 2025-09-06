from agent_dqn.algorithm.algorithm import Algorithm


class Agent:
    def __init__(self, device, monitor):
        self.algorithm = Algorithm(device, monitor)
        self.device = device
        self.monitor = monitor

    def predict(self, state):
        return self.algorithm.predict(state)

    def learn(self, experiences):
        self.algorithm.learn(experiences)

