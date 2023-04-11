import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

        # 参数初始化
        nn.init.uniform_(self.fc1.weight, -1/(state_dim**0.5), 1/(state_dim**0.5))
        nn.init.uniform_(self.fc1.bias, -1/(state_dim**0.5), 1/(state_dim**0.5))
        nn.init.uniform_(self.fc2.weight, -1/(hidden_dim_1**0.5), 1/(hidden_dim_1**0.5))
        nn.init.uniform_(self.fc2.bias, -1/(hidden_dim_1**0.5), 1/(hidden_dim_1**0.5))
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim_1, hidden_dim_2, action_dim):
        super(QValueNet, self).__init__()
        # 送入价值网络的是cat(s,a)
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc_out = torch.nn.Linear(hidden_dim_2, 1)

        # 参数初始化
        nn.init.uniform_(self.fc1.weight, -1/((state_dim + action_dim)**0.5), 1/((state_dim + action_dim)**0.5))
        nn.init.uniform_(self.fc1.bias, -1/((state_dim + action_dim)**0.5), 1/((state_dim + action_dim)**0.5))
        nn.init.uniform_(self.fc2.weight, -1 / (hidden_dim_1 ** 0.5), 1 / (hidden_dim_1 ** 0.5))
        nn.init.uniform_(self.fc2.bias, -1 / (hidden_dim_1 ** 0.5), 1 / (hidden_dim_1 ** 0.5))
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc_out.bias, -3e-3, 3e-3)

    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)