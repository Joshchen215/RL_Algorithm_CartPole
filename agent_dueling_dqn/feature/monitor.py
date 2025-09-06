import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Monitor:
    def __init__(self):
        self.loss_log = []
        self.epsilon_log = []
        self.reward_log = []
        self.episode_duration_log = []

    def reset(self):
        self.loss_log = []
        self.epsilon_log = []
        self.reward_log = []
        self.episode_duration_log = []

    def add_loss_info(self, loss):
        """
        向监视器添加新的loss信息
        """
        self.loss_log.append(loss)

    def add_epsilon_info(self, epsilon):
        """
        向监视器添加新的epsilon信息
        """
        self.epsilon_log.append(epsilon)

    def add_reward_info(self, reward):
        """
        向监视器添加新一轮episode的reward信息
        """
        self.reward_log.append(reward.item())

    def add_duration_info(self, duration):
        """
        向监视器添加新的epsilon信息
        """
        self.episode_duration_log.append(duration)

    def plot_loss(self, save_path=None):
        """
        绘制loss曲线
        """
        plt.figure()
        plt.plot(self.loss_log)
        plt.xlabel('train_count')
        plt.ylabel('loss')
        plt.title('TD error/ loss')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()
        else:
            plt.show()

    def plot_duration(self, save_path=None):
        """
        绘制回合长度曲线
        """
        plt.figure()
        plt.plot(self.episode_duration_log)
        plt.xlabel('episode')
        plt.ylabel('step')
        plt.title('Game Step Variation with Episode')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()
        else:
            plt.show()

    def plot_epsilon(self, save_path=None):
        """
        绘制epsilon曲线
        """
        plt.figure()
        plt.plot(self.epsilon_log)
        plt.xlabel('episode')
        plt.ylabel('epsilon')
        plt.title('Epsilon Variation with Episode')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()
        else:
            plt.show()

    def plot_reward(self, save_path=None):
        """
        绘制reward曲线
        """
        plt.figure()
        plt.plot(self.reward_log)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('Reward Variation with Episode')
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()
        else:
            plt.show()

    def plot_loss_and_duration(self, save_path=None):
        """
        在同一画布上绘制loss曲线、回合步长曲线和epsilon曲线
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 绘制loss曲线
        ax1.plot(self.loss_log)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('TD error/ loss')
        ax1.set_title('Loss Variation with Iteration')
        ax1.grid(True)

        # 绘制回合步长曲线
        ax2.plot(self.episode_duration_log)
        ax2.set_xlabel('episode')
        ax2.set_ylabel('step')
        ax2.set_title('Step Variation with Epsidoe')
        ax2.grid(True)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()
        else:
            plt.show()
