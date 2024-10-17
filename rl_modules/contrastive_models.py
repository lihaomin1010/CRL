import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# the flatten mlp
class flatten_mlp_contrastive(nn.Module):
    def __init__(self, obs_dims, goal_dims, hidden_size, repr_dims, action_dims):
        super(flatten_mlp_contrastive, self).__init__()
        self.fc1 = nn.Linear(obs_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sa_repr_layer = nn.Linear(hidden_size, repr_dims)

        self.fc3 = nn.Linear(goal_dims, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.g_repr_layer = nn.Linear(hidden_size, repr_dims)

        self.obs_dims = obs_dims

        self.apply(weights_init_)

    def forward(self, observation, goal, action=None):
        inputs_1 = torch.cat([observation, action], dim=1)
        inputs_2 = goal

        x = F.relu(self.fc1(inputs_1))
        x = F.relu(self.fc2(x))
        sa_repr = self.sa_repr_layer(x)

        y = F.relu(self.fc3(inputs_2))
        y = F.relu(self.fc4(y))
        g_repr = self.g_repr_layer(y)

        return sa_repr, g_repr


# define the policy network - tanh gaussian policy network
class tanh_gaussian_actor(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.apply(weights_init_)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))
    



# the flatten mlp
class flatten_mlp_contrastive_guassian(nn.Module):
    def __init__(self, obs_dims, goal_dims, hidden_size, repr_dims, action_dims, log_std_min, log_std_max):
        super(flatten_mlp_contrastive_guassian, self).__init__()
        self.fc1 = nn.Linear(obs_dims + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sa_repr_mean_layer = nn.Linear(hidden_size, repr_dims)
        self.sa_repr_log_std_layer = nn.Linear(hidden_size, repr_dims)

        self.fc3 = nn.Linear(goal_dims, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.g_repr_mean_layer = nn.Linear(hidden_size, repr_dims)
        self.g_repr_log_std_layer = nn.Linear(hidden_size, repr_dims)


        self.a = nn.Parameter(torch.randn(1)) # A scalar parameter
        self.b = nn.Parameter(torch.randn(1))  # Another scalar parameter
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.obs_dims = obs_dims
        self.apply(weights_init_)

    
    
    def forward(self, observation, goal, action=None):
        inputs_1 = torch.cat([observation, action], dim=1)
        inputs_2 = goal

        x = F.relu(self.fc1(inputs_1))
        x = F.relu(self.fc2(x))
        # sa_repr = self.sa_repr_layer(x)
        sa_repr_mean = self.sa_repr_mean_layer(x)
        sa_repr_log_std = self.sa_repr_log_std_layer(x)

        y = F.relu(self.fc3(inputs_2))
        y = F.relu(self.fc4(y))
        # g_repr = self.g_repr_layer(y)
        g_repr_mean = self.g_repr_mean_layer(y)
        g_repr_log_std = self.g_repr_log_std_layer(y)

        sa_repr_log_std = torch.clamp(sa_repr_log_std, min=self.log_std_min, max=self.log_std_max)
        g_repr_log_std = torch.clamp(g_repr_log_std, min=self.log_std_min, max=self.log_std_max) 

        # return sa_repr, g_repr
        # return sa_repr_mean, torch.exp(sa_repr_log_std, g_repr_mean, torch.exp(g_repr_log_std)
        return sa_repr_mean, sa_repr_log_std, g_repr_mean, g_repr_log_std
    

    def sample_latents(self, observation, goal, action=None):
        sa_repr_mean, sa_repr_log_std, g_repr_mean, g_repr_log_std = self.forward(observation, goal, action)
        sa_repr_std = torch.exp(sa_repr_log_std)
        g_repr_std = torch.exp(g_repr_log_std)
        
        # Sample a latent variable from the Gaussian distribution for both (s, a) and g
        z_sa = sa_repr_mean + sa_repr_std * torch.randn_like(sa_repr_mean)
        z_g = g_repr_mean + g_repr_std * torch.randn_like(g_repr_mean)

        return z_sa, z_g, sa_repr_mean, sa_repr_log_std, g_repr_mean, g_repr_log_std
    

    def calc_logits(self, l2_norms):
        return -self.a * l2_norms + self.b




class flatten_mlp_contrastive_img(nn.Module):
    def __init__(self, hidden_size, repr_dims, action_dims):
        super(flatten_mlp_contrastive_img, self).__init__()
        self.conv_net = AtariConvNet_Critic()

        self.fc1 = nn.Linear(1024 + action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sa_repr_layer = nn.Linear(hidden_size, repr_dims)

        self.fc3 = nn.Linear(1024, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.g_repr_layer = nn.Linear(hidden_size, repr_dims)

        self.apply(weights_init_)

    def forward(self, observation, goal, action=None):
        # Get the representation of the observation
        observation = self.conv_net(observation)
        observation = observation.reshape(observation.shape[0], -1)

        # Get the representation of the goal
        goal = self.conv_net(goal)
        goal = goal.reshape(goal.shape[0], -1)

        inputs_1 = torch.cat([observation, action], dim=1)
        inputs_2 = goal

        x = F.relu(self.fc1(inputs_1))
        x = F.relu(self.fc2(x))
        sa_repr = self.sa_repr_layer(x)

        y = F.relu(self.fc3(inputs_2))
        y = F.relu(self.fc4(y))
        g_repr = self.g_repr_layer(y)

        return sa_repr, g_repr



# define the policy network - tanh gaussian policy network
class tanh_gaussian_actor_img(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(tanh_gaussian_actor_img, self).__init__()
        self.conv_net = AtariConvNet_Actor()

        self.fc1 = nn.Linear(1024 * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)

        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
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

        mean = self.mean(x)
        log_std = self.log_std(x)

        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))


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


class AtariConvNet_Critic(nn.Module):
    def __init__(self):
        super(AtariConvNet_Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.apply(weights_init_)

    def forward(self, x):
        x = x.reshape(-1, 3, 64, 64)
        x = x.view(-1, 64, 64, 3)
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



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    obs = torch.randn(5, 3 * 64 * 64)
    goal = torch.randn(5, 3 * 64 * 64)
    action = torch.randn(5, 4)

    hidden_size = 256
    repr_dims = 64
    action_dims = 4
    
    model = flatten_mlp_contrastive_img(hidden_size, repr_dims, action_dims)
    sa_repr, g_repr = model(obs, goal, action)
    print(sa_repr.shape, g_repr.shape)
    print(sa_repr.shape)
    print(g_repr.shape)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    print(count_parameters(model))