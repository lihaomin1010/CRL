import numpy as np
import torch
import random

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = 0.6  # 优先级采样的指数
        self.beta = 0.4   # 重要性采样的指数
        self.beta_increment = 0.001  # beta的增量
        self.epsilon = 0.01  # 避免优先级为0
        
    def push(self, input_tensor, target_tensor):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
            self.priorities.append(None)
            
        # 新数据的优先级设为最大优先级
        # max_priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer[self.position] = (input_tensor, target_tensor)
        # self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size)
        
        # 获取样本
        samples = [self.buffer[idx] for idx in indices]
        input_tensors = torch.stack([s[0] for s in samples])
        target_tensors = torch.stack([s[1] for s in samples])

        return input_tensors, target_tensors, indices
        
    #def update_priorities(self, indices, priorities):
    #    for idx, priority in zip(indices, priorities):
    #        self.priorities[idx] = priority + self.epsilon
            
    def __len__(self):
        return len(self.buffer) 