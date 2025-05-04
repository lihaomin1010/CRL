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
from rl_modules.td3_img_agent import td3_img_agent
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
    if env_name == "fetch_reach_image":
        CLASS = FetchReachImage
        max_episode_steps = 50
    elif env_name == "fetch_push_image":
        CLASS = FetchPushImage
        max_episode_steps = 50
        kwargs["rand_y"] = True
    else:
        raise NotImplementedError("Unsupported environment: %s" % env_name)

    gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
    obs_dim = gym_env.observation_space.shape[0] // 2
    return gym_env, obs_dim, max_episode_steps





class FetchReachImage(reach.FetchReachEnv):
    """Wrapper for the FetchReach environment with image observations."""

    def __init__(self):
        self._dist = []
        self._dist_vec = []
        super(FetchReachImage, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 9), 0),
            high=np.full((64 * 64 * 9), 255),
            dtype=np.uint8,
        )
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def reset(self):
        if self._dist:  # if len(self._dist) > 0, ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s = super(FetchReachImage, self).reset()
        self.observation_space = self._new_observation_space
        self._goal = s["desired_goal"].copy()

        for _ in range(10):
            hand = s["achieved_goal"]
            obj = s["desired_goal"]
            delta = obj - hand
            a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
            s, _, _, _ = super(FetchReachImage, self).step(a)

        self._goal_img = self.observation(s)

        self.observation_space = self._old_observation_space
        s = super(FetchReachImage, self).reset()
        self.observation_space = self._new_observation_space
        img = self.observation(s)
        dist = np.linalg.norm(s["achieved_goal"] - self._goal)
        self._dist.append(dist)
        return np.concatenate([img, self._goal_img, img])

    def step(self, action):
        s, _, _, _ = super(FetchReachImage, self).step(action)
        dist = np.linalg.norm(s["achieved_goal"] - self._goal)
        self._dist.append(dist)
        done = False
        r = float(dist < 0.05)
        info = {}
        img = self.observation(s)
        return np.concatenate([img, self._goal_img, img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchReachImage, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
        self.viewer.cam.distance = 0.8
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -30





class FetchPushImage(push.FetchPushEnv):
    """Wrapper for the FetchPush environment with image observations."""

    def __init__(self, camera="camera2", start_at_obj=True, rand_y=False):
        self._start_at_obj = start_at_obj
        self._rand_y = rand_y
        self._camera_name = camera
        self._dist = []
        self._dist_vec = []
        super(FetchPushImage, self).__init__()
        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64 * 64 * 9), 0),
            high=np.full((64 * 64 * 9), 255),
            dtype=np.uint8,
        )
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
        self._dist_vec = []
        self._dist = []

    def _move_hand_to_obj(self):
        s = super(FetchPushImage, self)._get_obs()
        for _ in range(100):
            hand = s["observation"][:3]
            obj = s["achieved_goal"] + np.array([-0.02, 0.0, 0.0])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.06:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            s, _, _, _ = super(FetchPushImage, self).step(a)

    def reset(self):
        if self._dist:  # if len(self._dist) > 0 ...
            self._dist_vec.append(self._dist)
        self._dist = []

        # generate the new goal image
        self.observation_space = self._old_observation_space
        s = super(FetchPushImage, self).reset()
        self.observation_space = self._new_observation_space
        # Randomize object position
        for _ in range(8):
            super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        if not self._rand_y:
            object_qpos[1] = 0.75
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self._move_hand_to_obj()
        self._goal_img = self.observation(s)
        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3]
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            # print('Bad reset, recursing.')
            return self.reset()
        self._goal = block_xyz[:2].copy()

        self.observation_space = self._old_observation_space
        s = super(FetchPushImage, self).reset()
        self.observation_space = self._new_observation_space
        for _ in range(8):
            super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        object_qpos[:2] = np.array([1.15, 0.75])
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        if self._start_at_obj:
            self._move_hand_to_obj()
        else:
            for _ in range(5):
                super(FetchPushImage, self).step(self.action_space.sample())

        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz[:2] - self._goal)
        self._dist.append(dist)
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print("Bad reset, recursing.")
            return self.reset()
        return np.concatenate([img, self._goal_img, img])

    def step(self, action):
        s, _, _, _ = super(FetchPushImage, self).step(action)
        block_xy = self.sim.data.get_joint_qpos("object0:joint")[:2]
        dist = np.linalg.norm(block_xy - self._goal)
        self._dist.append(dist)
        done = False
        r = float(dist < 0.05)  # Taken from the original task code.
        info = {}
        img = self.observation(s)
        return np.concatenate([img, self._goal_img, img]), r, done, info

    def observation(self, observation):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=64, width=64)
        return img.flatten()

    def _viewer_setup(self):
        super(FetchPushImage, self)._viewer_setup()
        if self._camera_name == "camera1":
            self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
            self.viewer.cam.distance = 0.9
            self.viewer.cam.azimuth = 180
            self.viewer.cam.elevation = -40
        elif self._camera_name == "camera2":
            self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
            self.viewer.cam.distance = 0.65
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -40
        else:
            raise NotImplementedError


def get_env_params(env, max_episode_steps):
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
    td3_trainer = td3_img_agent(args, env, env_params)
    td3_trainer.learn()



if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
  
  

