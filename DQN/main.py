import torch
import gym
import random
import numpy as np
from DQN import DQN
import matplotlib.pyplot as plt
import argparse
import rl_utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v0', help="环境名称")
    parser.add_argument("--render_mode", default='human', help="渲染方式")
    parser.add_argument("--critic_lr", default=2e-3, help="价值网络学习率")
    parser.add_argument("--hidden_dim", default=128, help="隐藏层参数")
    parser.add_argument("--gamma", default=0.98, help="奖励折扣因子")
    parser.add_argument("--tau", default=0.001, help="目标网络软更新参数")
    parser.add_argument("--buffer_size", default=10000, help="回放池大小")
    parser.add_argument("--batch_size", default=20, help="每次训练采样的批次大小")
    parser.add_argument("--num_episodes", default=500, help="训练轮数")
    parser.add_argument("--minimal_size", default=500, help="回放池最小有这些样本才开始采样")
    parser.add_argument("--target_update", default=10, help="目标网络更新频次")
    parser.add_argument("--epsilon", default=0.1, help="贪婪函数概率")
    parser.add_argument("--sigma", default=0.01, help="高斯噪声标准差，均值为0")
    args = parser.parse_args()

# 设置随机数种子，便于结果复现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# 创建环境
env = gym.make(args.env_name, render_mode=args.render_mode)
# 观测空间维度，动作空间维度，动作最大值
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# 设置回放池
replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)
agent = DQN(state_dim, args.hidden_dim, action_dim, args.critic_lr, args.gamma, args.epsilon, args.target_update, device)
return_list = rl_utils.train_off_policy_agent(env, agent, args.num_episodes, replay_buffer, args.minimal_size, args.batch_size)
# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(args.env_name))
plt.savefig("./"+"DQN" + ".png")
plt.show()







