import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from her_modules.replay_buffer import ReplayBuffer


class CNet(nn.Module):
    def __init__(self, s_features, out_features):
        super(CNet, self).__init__()
        self.fc1 = nn.Linear(s_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        embeds = self.out(x)
        return embeds


class CRL:
    def __init__(self, s_features, out_features, learning_rate=0.001, buffer_size=10000, n_updates=1):
        self.s_features = s_features
        self.out_features = out_features
        self.lr = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_net = CNet(s_features * 2, out_features).to(self.device)

        self.n_updates = n_updates  # 每次train时的更新次数
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.CELoss = nn.CrossEntropyLoss()


        # 优化器
        self.optimizer = optim.Adam(self.train_net.parameters(), lr=self.lr)



    def train(self, input_tensor, score):
        # 将新数据存入replay buffer
        self.replay_buffer.push(input_tensor, score)

        total_loss = 0
        total_reward = None

        # 进行多次更新
        for _ in range(self.n_updates):
            # 从buffer中采样
            batch = self.replay_buffer.sample(batch_size=16)
            if batch is None:
                return None

            batch_input, scores, indices = batch

            r_t = torch.rand(batch_input.shape).to(self.device)
            batch_input_1 = torch.cat([batch_input, r_t],dim=2)

            r_t = torch.rand(batch_input.shape).to(self.device)
            batch_input_2 = torch.cat([batch_input, r_t],dim=2)

            scores_weight =  (scores - scores.transpose(1,2)).pow(2)
            scores_weight = scores_weight / scores_weight.max()

            scores_dim = scores.squeeze(dim=2)
            scores_dis = scores_dim.max(dim=1).values - scores_dim.min(dim=1).values
            # 前向传播
            x_1 = self.train_net(batch_input_1)
            x_2 = self.train_net(batch_input_2)

            sim = torch.einsum('xyz, xwz -> xyw', x_1, x_2)
            sim = torch.sigmoid(sim*5)

            I = torch.eye(sim.shape[1], device=self.device).unsqueeze(0).expand(sim.shape[0], -1, -1)

            loss = scores_weight*torch.pow(sim-I, 2)

            #loss = torch.sum(scores_dis*torch.mean(loss, dim=(1,2))).mean()
            loss = torch.sum(torch.mean(loss, dim=(1, 2))).mean()


            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.n_updates


    def get_skill_entropy(self, input_tensor):
        r_t = torch.rand(input_tensor.shape).to(self.device)
        input = torch.cat([input_tensor, r_t],dim=1).to(self.device)
        x = self.train_net(input)

        dist = torch.einsum("ik,jk->ij", x, x)

        return dist.mean(dim=1).unsqueeze(1)










