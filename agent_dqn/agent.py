import torch
from agent_dqn.algorithm.algorithm import Algorithm


class Agent:
    def __init__(self, device, monitor):
        self.algorithm = Algorithm(device, monitor)
        self.device = device
        self.monitor = monitor

    def predict(self, state, exploit_flag=False):
        return self.algorithm.predict(state, exploit_flag)

    def learn(self, experiences):
        self.algorithm.learn(experiences)

    def save_model(self, path=None, id="1"):
        # 保存模型
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        print(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # 加载模型
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        print(f"load model {model_file_path} successfully")

