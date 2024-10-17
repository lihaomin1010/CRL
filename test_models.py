import numpy as np
from rl_modules.equ_constrastive_models import equi_flatten_mlp_contrastive_block_reach


obs_dims = 10
goal_dims = 10
hidden_size = 128
repr_dims = 64
action_dims = 4
N = 64

model = equi_flatten_mlp_contrastive_block_reach(
    obs_dims, goal_dims, hidden_size, repr_dims, action_dims, initialize=True, N=N
)


if name == "main":
    batch_size = 10
    observation = torch.randn(batch_size, obs_dims)
    goal = torch.randn(batch_size, goal_dims)
    action = torch.rand(batch_size, action_dims)
