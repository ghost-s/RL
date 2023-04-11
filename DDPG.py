import numpy as np
import torch
import torch.nn.functional as F

from actor_critic_network import PolicyNet, QValueNet

save_actor_path = 'D:/MADDPG/DDPG/actor.pth'
save_target_actor_path = 'D:/MADDPG/DDPG/target_actor.pth'
save_critic_path = 'D:/MADDPG/DDPG/critic.pth'
save_target_critic_path = 'D:/MADDPG/DDPG/target_critic.pth'


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        # 定义价值网络 策略网络 目标价值网络 目标策略网络
        self.actor = PolicyNet(state_dim, hidden_dim_1, hidden_dim_2, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim_1, hidden_dim_2, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim_1, hidden_dim_2, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim_1, hidden_dim_2, action_dim).to(device)

        # 载入预训练权重(没有预训练权重时，目标价值网络与目标策略网络载入与价值网络和策略网络相同的参数初始化)
        # self.target_critic.load_state_dict(torch.load(save_target_critic_path))
        # self.critic.load_state_dict(torch.load(save_critic_path))
        # self.target_actor.load_state_dict(torch.load(save_target_actor_path))
        # self.actor.load_state_dict(torch.load(save_actor_path))
        # 没有预训练权重时，初始化目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 定义价值网络与策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-2)

        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # 策略网络输出动作
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索，提升网络的鲁棒性
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    # 软更新目标网络的参数
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # 更新网络参数
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # TD算法更新价值网络
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新策略网络
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络参数
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    # 保存模型参数
    def save_models(self):
        torch.save(self.actor.state_dict(), save_actor_path)
        torch.save(self.target_actor.state_dict(), save_target_actor_path)
        torch.save(self.critic.state_dict(), save_critic_path)
        torch.save(self.target_critic.state_dict(), save_target_critic_path)
