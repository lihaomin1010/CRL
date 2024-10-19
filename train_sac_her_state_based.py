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
from rl_modules.sac_agent import sac_agent
import random
import torch


from scipy.spatial.transform import Rotation as R



os.environ['SDL_VIDEODRIVER'] = 'dummy'


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat

def load(env_name):
  """Loads the train and eval environments, as well as the obs_dim."""
  # pylint: disable=invalid-name
  kwargs = {}
  if env_name == 'fetch_reach':
    CLASS = FetchReachEnv
    max_episode_steps = 50
  elif env_name == 'fetch_push':
    # CLASS = FetchPushEnv
    CLASS = FetchPushEnv2
    max_episode_steps = 50
  elif env_name == 'fetch_pick_and_place':
    CLASS = FetchPickAndPlace
    max_episode_steps = 50
  elif env_name == 'sawyer_push':
    CLASS = SawyerPush
    max_episode_steps = 150
  elif env_name == 'sawyer_bin':
    CLASS = SawyerBin
    max_episode_steps = 150
  elif env_name == 'sawyer_drawer':
    CLASS = SawyerDrawer
    max_episode_steps = 150
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)
  
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  return gym_env, obs_dim, max_episode_steps




def convert_xyz_to_zyx(xyz_angles):
    """
    Convert Euler angles from XYZ convention to ZYX convention.
    
    :param xyz_angles: A tuple or list of three angles (in radians) in the XYZ convention.
    :return: A tuple of three angles (in radians) in the ZYX convention.
    """
    # Step 1: Create a rotation object from the XYZ Euler angles
    rotation_xyz = R.from_euler('XYZ', xyz_angles)

    # Step 2: Convert the rotation to ZYX Euler angles
    zyx_angles = rotation_xyz.as_euler('ZYX')

    return zyx_angles


    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[Ellipsis, 0] = cj * cc + sj * ss
    quat[Ellipsis, 3] = cj * sc - sj * cs
    quat[Ellipsis, 2] = -(cj * ss + sj * cc)
    quat[Ellipsis, 1] = cj * cs - sj * sc
    return quat


class FetchReachEnv(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment."""

  def __init__(self):
    super(FetchReachEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((30,), -np.inf),
        high=np.full((30,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchReachEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchReachEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)  # Default from Fetch environment.
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 0
    end_index = 3
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    ag = np.zeros_like(s)
    ag[start_index:end_index] = observation['achieved_goal']
    
    #return {'observation':s,'desired_goal':g,'achieved_goal':ag}
    observation = np.concatenate([s, g, ag], axis=0)
    # observation = np.concatenate([s, g, s], axis=0)
    return observation


class FetchPushEnv(push.FetchPushEnv):
  """Wrapper for the FetchPush environment."""

  def __init__(self):
    super(FetchPushEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((75,), -np.inf),
        high=np.full((75,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    ag = np.zeros_like(s)
    ag[start_index:end_index] = observation['achieved_goal']
    
    #return {'observation':s,'desired_goal':g,'achieved_goal':ag}
    observation = np.concatenate([s, g, ag], axis=0)
    # observation = np.concatenate([s, g, s], axis=0)
    return observation
  



class FetchPushEnv2(push.FetchPushEnv):
  """Wrapper for the FetchPush environment."""

  def __init__(self):
    super(FetchPushEnv2, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((78,), -np.inf),
        high=np.full((78,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv2, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv2, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    
    
    s = observation['observation']
    
    ee_xyz = s[0:3]
    block_xyz = s[3:6]
    relative_diff_xyz = s[6:9]
    joint_right_finger_disp = s[9:10]
    joint_left_finger_disp = s[10:11]

    obj_orientation_euler_xyz = s[11:14]

    relative_linear_vel_xyz = s[14:17]
    angular_vel_xyz = s[17:20]
    ee_vel_xyz = s[20:23]
    joint_right_finger_vel = s[23:24]
    joint_left_finger_vel = s[24:25]


    # Convert the angle z of obj_orientation_euler to [cos(angle_z), sin(angle_z)], and then concatenate everything
    # except the last element of obj_orientation_euler i.e. angle_z, to make the new state of size 26
    obj_orientation_euler_zyx = convert_xyz_to_zyx(obj_orientation_euler_xyz)

    obj_orientation_euler_z = obj_orientation_euler_zyx[0]
    obj_orientation_euler_y = obj_orientation_euler_zyx[1]
    obj_orientation_euler_x = obj_orientation_euler_zyx[2]

    cos_obj_orientation_euler_z = np.cos(obj_orientation_euler_z)
    sin_obj_orientation_euler_z = np.sin(obj_orientation_euler_z)

    new_obj_orientation = np.array([obj_orientation_euler_x, obj_orientation_euler_y,cos_obj_orientation_euler_z, sin_obj_orientation_euler_z])


    new_state = np.concatenate([ee_xyz, 
                                block_xyz, 
                                relative_diff_xyz, 
                                joint_right_finger_disp, 
                                joint_left_finger_disp, 
                                new_obj_orientation, 
                                relative_linear_vel_xyz, 
                                angular_vel_xyz, 
                                ee_vel_xyz, 
                                joint_right_finger_vel, 
                                joint_left_finger_vel])


    s = new_state
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    ag = np.zeros_like(s)
    ag[start_index:end_index] = observation['achieved_goal']
    
    #return {'observation':s,'desired_goal':g,'achieved_goal':ag}
    observation = np.concatenate([s, g, ag], axis=0)
    # observation = np.concatenate([s, g, s], axis=0)
    return observation
  

class FetchPickAndPlace(pick_and_place.FetchPickAndPlaceEnv):
  """Wrapper for the FetchPush environment."""

  def __init__(self):
    super(FetchPickAndPlace, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((78,), -np.inf),
        high=np.full((78,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPickAndPlace, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPickAndPlace, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    
    
    s = observation['observation']
    
    ee_xyz = s[0:3]
    block_xyz = s[3:6]
    relative_diff_xyz = s[6:9]
    joint_right_finger_disp = s[9:10]
    joint_left_finger_disp = s[10:11]

    obj_orientation_euler = s[11:14]

    relative_linear_vel_xyz = s[14:17]
    angular_vel_xyz = s[17:20]
    ee_vel_xyz = s[20:23]
    joint_right_finger_vel = s[23:24]
    joint_left_finger_vel = s[24:25]


    # Convert the angle z of obj_orientation_euler to [cos(angle_z), sin(angle_z)], and then concatenate everything
    # except the last element of obj_orientation_euler i.e. angle_z, to make the new state of size 26

    obj_orientation_euler_z = obj_orientation_euler[2]
    cos_obj_orientation_euler_z = np.cos(obj_orientation_euler_z)
    sin_obj_orientation_euler_z = np.sin(obj_orientation_euler_z)

    new_obj_orientation = np.array([obj_orientation_euler[0], obj_orientation_euler[1], cos_obj_orientation_euler_z, sin_obj_orientation_euler_z])

    new_state = np.concatenate([ee_xyz, 
                                block_xyz, 
                                relative_diff_xyz, 
                                joint_right_finger_disp, 
                                joint_left_finger_disp, 
                                new_obj_orientation, 
                                relative_linear_vel_xyz, 
                                angular_vel_xyz, 
                                ee_vel_xyz, 
                                joint_right_finger_vel, 
                                joint_left_finger_vel])


    s = new_state
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    ag = np.zeros_like(s)
    ag[start_index:end_index] = observation['achieved_goal']
    
    observation = np.concatenate([s, g, ag], axis=0)
    return observation
  



class SawyerPush(metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for the SawyerPush environment."""

  def __init__(self,
               goal_min_x=-0.1,
               goal_min_y=0.5,
               goal_max_x=0.1,
               goal_max_y=0.9):
    
    super(SawyerPush, self).__init__()
    self._random_reset_space.low[3] = goal_min_x
    self._random_reset_space.low[4] = goal_min_y
    self._random_reset_space.high[3] = goal_max_x
    self._random_reset_space.high[4] = goal_max_y
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(21, -np.inf),
        high=np.full(21, np.inf),
        dtype=np.float32)
        

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    gripper_distance = np.linalg.norm(finger_right - finger_left)
    gripper_distance = np.clip(gripper_distance / 0.1, 0., 1.)
    obj = self._get_pos_objects()
    # Note: we should ignore the target gripper distance. The arm goal is set
    # to be the same as the puck goal.
    state = np.concatenate([tcp_center, obj, [gripper_distance]])
    goal = np.concatenate([self._target_pos, self._target_pos, [0.5]])
    # ag = np.concatenate([self._get_pos_objects(), self._get_pos_objects(), [0.5]])

    return np.concatenate([state, goal,state]).astype(np.float32)

  
  def step(self, action):
    obs = super(SawyerPush, self).step(action)
    dist = np.linalg.norm(self._target_pos - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from the metaworld code.
    return obs, r, False, {}





class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self):
    self._goal = np.zeros(3)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    t = np.random.random()
    self._goal = t * pos1 + (1 - t) * pos2
    self._goal[2] = np.random.uniform(0.03, 0.12)
    return self._get_obs()

  def step(self, action):
    super(SawyerBin, self).step(action)
    dist = np.linalg.norm(self._goal - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand,self._get_pos_objects(),
                          [gripper_distance_apart]))
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           self._goal, [0.4]])
    

    return np.concatenate([obs, goal,obs]).astype(np.float32)
  

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(3 * 7, -np.inf),
        high=np.full(3
                      * 7, np.inf),
        dtype=np.float32)
        
  



class SawyerDrawer(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['drawer-close-v2']):
  """Wrapper for the SawyerDrawer environment."""

  def __init__(self):
    super(SawyerDrawer, self).__init__()
    self._random_reset_space.low[0] = 0
    self._random_reset_space.high[0] = 0
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._target_pos = np.zeros(0)  # We will overwrite this later.
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def _get_pos_objects(self):
    return self.get_body_com('drawer_link') +  np.array([.0, -.16, 0.0])

  def reset_model(self):
    super(SawyerDrawer, self).reset_model()
    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    self._target_pos = self._get_pos_objects().copy()

    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    return self._get_obs()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(3 * 4, -np.inf),
        high=np.full(3 * 4, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    obj = self._get_pos_objects()

    # Arm position is same as drawer position. We only provide the drawer Y coordinate.
    return np.concatenate([tcp_center, [obj[1]],self._target_pos, [self._target_pos[1]], tcp_center, [obj[1]]])
    # return np.concatenate([tcp_center, [obj[0],obj[1]],self._target_pos, [self._target_pos[0],self._target_pos[1]], tcp_center, [obj[0],obj[1]]])

  def step(self, action):
    obs = super(SawyerDrawer, self).step(action)
    return obs, 0.0, False, {}
        


def get_env_params(env,max_episode_steps):
    obs = env.reset()
    params = {
        "obs": int(obs.shape[0] / 3),
        "goal": int(obs.shape[0] / 3),
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
    }
    params["max_timesteps"] = max_episode_steps
    return params


def launch(args):
    if args.seed == 123:
      seed  = np.random.randint(0,10000)
    else:
       seed = args.seed
      
    print("Seed: ", seed)
    print("GPU: ", args.cuda)



    # create the contrastive_agent
    env, obs_dim, max_episode_steps = load(args.env_name)

    # set random seeds for reproduce
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get the environment parameters    
    env_params = get_env_params(env,max_episode_steps)
    # create the sac agent to interact with the environment 
    sac_trainer = sac_agent(args, env, env_params)
    sac_trainer.learn()



if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
  
  

