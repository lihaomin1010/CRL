import gym
from gym import spaces
from gym.vector import AsyncVectorEnv
from gym.envs.robotics.fetch import push
from gym.envs.robotics.fetch import reach
from gym.envs.robotics.fetch import pick_and_place
import metaworld

import numpy as np
import gym
import os, sys

from arguments_sac import get_args
from mpi4py import MPI
from rl_modules.td3_agent_bulletarm import td3_agent
import random
import torch
from bulletarm import env_factory


from scipy.spatial.transform import Rotation as R



def get_env_params(env,max_episode_steps):
    obs = env.reset()
    # actions = env.getNextAction()
    # close the environment
    params = {'obs': int(obs[2].shape[1] / 3),
            'goal': int(obs[2].shape[1] / 3),
            'action': 5,
            'action_max': 1.0,
            'action_ranges': np.array([[0,-0.05,-0.05,-0.05,-np.pi/16],
                              [1,0.05,0.05,0.05,np.pi/16]])
            }
    params['max_timesteps'] = max_episode_steps
    print(params)

    return params

    

def launch(args):
    # create the contrastive_agent
    num_envs = 1
    env_config = {'render': False, 'random_orientation': True,'obs_type': "vector",'action_sequence':'pxyzr'}

    if args.seed == 123:
      seed  = np.random.randint(0,10000)
    else:
       seed = args.seed

    if args.env_name == "block_reach":
        print("Block Reach")
        envs = env_factory.createEnvs(num_envs, 'close_loop_block_reaching_goal', env_config)
    elif args.env_name == "block_push":
        print("Block Push")
        envs = env_factory.createEnvs(num_envs, 'close_loop_block_pushing_goal', env_config)
    elif args.env_name == "block_pick":
        print("Block Pick")
        envs = env_factory.createEnvs(num_envs, 'close_loop_block_picking_goal', env_config)
    elif args.env_name == "block_pick":
        print("Block Pick")
        envs = env_factory.createEnvs(num_envs, 'close_loop_block_picking_goal', env_config)
    elif args.env_name == "block_pick_and_place":
        print("Block Pick and Place")
        envs = env_factory.createEnvs(num_envs, 'close_loop_block_in_bowl_goal', env_config)

    print("Environments Created")
    
    # get the params
    env_params = get_env_params(envs,50)

    # set random seeds for reproduction
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env_params = env_params = get_env_params(envs,50)
    
    # create the ddpg agent to interact with the environment 
    td3_trainer = td3_agent(args, envs, env_params)
    td3_trainer.learn()



if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
  
  

