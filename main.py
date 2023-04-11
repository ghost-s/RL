import torch
import rl_utils
import gym
import numpy as np
from DDPG import DDPG
import random
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='Pendulum-v1', help="环境名称")
    parser.add_argument("--render_mode", default='human', help="渲染方式")
    parser.add_argument("--actor_lr", default=3.5e-4, help="价值网络学习率")
    parser.add_argument("--critic_lr", default=1.2e-3, help="策略网络学习率")
    parser.add_argument("--hidden_dim_1", default=400, help="隐藏层1参数")
    parser.add_argument("--hidden_dim_2", default=300, help="隐藏层2参数")
    parser.add_argument("--gamma", default=0.99, help="奖励折扣因子")
    parser.add_argument("--tau", default=0.001, help="目标网络软更新参数")
    parser.add_argument("--buffer_size", default=1000000, help="回放池大小")
    parser.add_argument("--batch_size", default=256, help="每次训练采样的批次大小")
    parser.add_argument("--num_episodes", default=200, help="训练轮数")
    parser.add_argument("--minimal_size", default=256, help="回放池最小有这些样本才开始采样")
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
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
# 设置回放池
replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)

agent = DDPG(state_dim, args.hidden_dim_1, args.hidden_dim_2, action_dim, action_bound, args.sigma, args.actor_lr, args.critic_lr, args.tau, args.gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, args.num_episodes, replay_buffer, args.minimal_size, args.batch_size)
# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(args.env_name))
plt.savefig("./"+"DDPG" + ".png")
plt.show()


