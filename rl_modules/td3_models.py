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







# define the policy network - tanh gaussian policy network
class deterministic_actor_img(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size):
        super(deterministic_actor_img, self).__init__()
        self.conv_net = AtariConvNet_Actor()

        self.fc1 = nn.Linear(1024 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.action_out = nn.Linear(hidden_size, action_dims)

        # the log_std_min and log_std_max
        self.obs_dims = int(input_dims / 2)
        self.apply(weights_init_)

    def forward(self, obs):
        obs = obs.squeeze()
        try:
            observation = obs[:, : self.obs_dims]
        except:
            observation = obs[: self.obs_dims]
        observation = observation.view(-1, 64, 64, 3)
        observation = observation.permute(0, 3, 1, 2)

        # o1 = (observation[0] * 255).type(torch.uint8)
        # print("observation shape-2:", o1.shape)
        # print(torch.unique(o1))
        # plt.imshow((o1).cpu().permute(1, 2, 0))
        # plt.show()

        try:
            goal = obs[:, self.obs_dims :]
        except:
            goal = obs[self.obs_dims :]
        goal = goal.view(-1, 64, 64, 3)
        goal = goal.permute(0, 3, 1, 2)

        # g = (goal[0] * 255).type(torch.uint8)
        # print("observation shape-2:", g.shape)
        # print(torch.unique(g))
        # plt.imshow((g).cpu().permute(1, 2, 0))
        # plt.show()

        obs = self.conv_net(observation)
        obs = obs.reshape(obs.shape[0], -1)

        goal = self.conv_net(goal)
        goal = goal.reshape(goal.shape[0], -1)

        x = torch.cat([obs, goal], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.action_out(x)

        return mean
    




class AtariConvNet_Actor(nn.Module):
    def __init__(self):
        super(AtariConvNet_Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x



    


class flatten_mlp_img(nn.Module):
    def __init__(self, hidden_size, action_dims):
        super(flatten_mlp_img, self).__init__()
        self.conv_net = AtariConvNet_Critic()

        self.fc1 = nn.Linear((1024 * 2) + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value_layer = nn.Linear(hidden_size, 1)


        self.apply(weights_init_)

    def forward(self, observation_goal,action=None):
        observation = observation_goal[:, : observation_goal.shape[1] // 2]
        goal = observation_goal[:, observation_goal.shape[1] // 2 :]

        # Get the representation of the observation
        observation = self.conv_net(observation)
        observation = observation.reshape(observation.shape[0], -1)

        # Get the representation of the goal
        goal = self.conv_net(goal)
        goal = goal.reshape(goal.shape[0], -1)

        input = torch.cat([observation, goal, action], dim=1)

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        q_value = self.q_value_layer(x)
        return q_value    


class AtariConvNet_Critic(nn.Module):
    def __init__(self):
        super(AtariConvNet_Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.apply(weights_init_)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 64, 64)
        x = x.view(x.shape[0], 64, 64, 3)
        x = x.permute(0, 3, 1, 2)

        # x1 = (x[0] * 255).type(torch.uint8)
        # print("observation shape-2:", x1.shape)
        # print(torch.unique(x1))
        # plt.imshow((x1).cpu().permute(1, 2, 0))
        # plt.show()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x