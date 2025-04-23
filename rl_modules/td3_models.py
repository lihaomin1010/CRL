import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""
# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


# define the policy network - tanh gaussian policy network
class deterministic_actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size):
        super(deterministic_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_out = nn.Linear(hidden_size, action_dims)

        self.apply(weights_init_)
        

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        
        actions = torch.tanh(self.action_out(x))
        return  actions


# the flatten mlp
class flatten_mlp(nn.Module):
    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(flatten_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size) if action_dims is None else nn.Linear(input_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

        self.apply(weights_init_)

    def forward(self, obs, action=None):
        inputs = torch.cat([obs, action], dim=1) if action is not None else obs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output
