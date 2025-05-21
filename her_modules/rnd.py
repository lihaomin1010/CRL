import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def create_nn(input_num, output_num, init_val=0.001, relu=True, trainable=True, name=''):
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.linear = nn.Linear(input_num, output_num)
            self.relu = nn.ReLU() if relu else nn.Identity()

            # 初始化权重
            nn.init.uniform_(self.linear.weight, -init_val, init_val)
            nn.init.uniform_(self.linear.bias, -init_val, init_val)

            if not trainable:
                for param in self.parameters():
                    param.requires_grad = False

        def forward(self, x):
            return self.relu(self.linear(x))

    return SimpleNN()


class RND:
    def __init__(self, s_features, out_features=3, name="", learning_rate=0.001):
        self.s_features = s_features
        self.out_features = out_features
        self.lr = learning_rate
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_net()

    def _build_net(self):
        # 创建训练网络
        self.train_net = nn.Sequential(
            create_nn(self.s_features, 64, relu=True, trainable=True, name='l1'),
            create_nn(64, self.out_features, relu=False, trainable=True, name='output')
        ).to(self.device)

        # 创建目标网络
        self.target_net = nn.Sequential(
            create_nn(self.s_features, 64, init_val=10, relu=True, trainable=False, name='l1'),
            create_nn(64, self.out_features, init_val=10, relu=False, trainable=False, name='output')
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.train_net.parameters(), lr=self.lr)

    def train(self, state):

        # 前向传播
        train_output = self.train_net(state)
        with torch.no_grad():
            target_output = self.target_net(state)

        # 计算损失
        loss = torch.mean((train_output - target_output) ** 2)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_intrinsic_reward(self, state):
        with torch.no_grad():
            train_output = self.train_net(state)
            target_output = self.target_net(state)
            reward = torch.mean((train_output - target_output) ** 2, dim=1)
        return reward.reshape((-1, 1))

    def get_target(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            target_output = self.target_net(state)
        return target_output.cpu().numpy()









