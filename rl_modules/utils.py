import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions import Distribution

"""
the tanhnormal distributions from rlkit may not stable
"""


class tanh_normal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(
            self.normal_mean.size(),
            dtype=torch.float32,
            device="cuda" if self.cuda else "cpu",
        )
        sample_std = torch.ones(
            self.normal_std.size(),
            dtype=torch.float32,
            device="cuda" if self.cuda else "cpu",
        )
        z = (
            self.normal_mean
            + self.normal_std * Normal(sample_mean, sample_std).sample()
        )
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


# get action_infos
class get_action_info:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = tanh_normal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)


# env wrapper
class env_wrapper:
    def __init__(self, env, args):
        self._env = env
        self.args = args
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        self.timesteps = 0
        obs = self._env.reset()
        return obs

    def step(self, action):
        # revise the correct action range
        obs, reward, done, info = self._env.step(action)
        # increase the timesteps
        self.timesteps += 1
        if self.timesteps >= self.args.episode_length:
            done = True
        return obs, reward, done, info

    def render(self):
        """
        to be Implemented during execute the demo
        """
        self._env.render()

    def seed(self, seed):
        """
        set environment seeds
        """
        self._env.seed(seed)


# record the reward info of the dqn experiments
class reward_recorder:
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        if self.get_length >= self.history_length:
            self.buffer.pop(0)
        # append new one
        self.buffer.append(0.0)
        self._episode_length += 1

    # get length of buffer
    @property
    def get_length(self):
        return len(self.buffer)

    @property
    def mean(self):
        return np.mean(self.buffer)

    # get the length of total episodes
    @property
    def num_episodes(self):
        return self._episode_length


def perturbVec_fetch_push_pick(obs, next_obs, action, goal, random_goal):
    aug_obs = obs.copy()
    aug_next_obs = next_obs.copy()
    aug_action = action.copy()
    aug_goal = goal.copy()
    aug_random_goal = random_goal.copy()

    theta = np.random.random() * 2 * np.pi - np.pi
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotate the first 2 elements of the action i.e. x,y of the end effector
    aug_action[0:2] = rot.dot(aug_action[0:2])

    # Rotate the Observation, Next Observation, Goal, Random Goal
    vec_dict = {
        "obs": obs,
        "next_obs": next_obs,
        "goal": goal,
        "random_goal": random_goal,
    }

    # print("Aug Obs: ", aug_obs)
    # print("Aug Next Obs: ", aug_next_obs)
    # print("Aug Action: ", aug_action)
    # print("Aug Goal: ", aug_goal)
    # print("Aug Random Goal: ", aug_random_goal)

    # Rotate the Observation, Next Observation, Goal, Random Goal
    for key, vec in vec_dict.items():
        vec_ee_xy = vec[0:2]
        vec_ee_z = vec[2:3]

        vec_block_xy = vec[3:5]
        vec_block_z = vec[5:6]

        vec_relative_diff_xy = vec[6:8]
        vec_relative_diff_z = vec[8:9]

        vec_joint_right_finger_disp = vec[9:10]
        vec_joint_left_finger_disp = vec[10:11]

        vec_obj_orientation_euler_xy = vec[11:13]
        obj_orientation_euler_cos_sin_z = vec[13:15]

        vec_relative_linear_vel_xy = vec[15:17]
        vec_relative_linear_vel_z = vec[17:18]

        vec_angular_vel_xy = vec[18:20]
        vec_angular_vel_z = vec[20:21]

        vec_ee_vel_xy = vec[21:23]
        vec_ee_vel_z = vec[23:24]

        vec_joint_right_finger_vel = vec[24:25]
        vec_joint_left_finger_vel = vec[25:26]

        aug_vec_ee_xy = rot.dot(vec_ee_xy)
        aug_vec_block_xy = rot.dot(vec_block_xy)
        aug_vec_relative_diff_xy = rot.dot(vec_relative_diff_xy)
        aug_obj_orientation_euler_cos_sin_z = rot.dot(obj_orientation_euler_cos_sin_z)
        aug_vec_relative_linear_vel_xy = rot.dot(vec_relative_linear_vel_xy)
        aug_vec_angular_vel_xy = rot.dot(vec_angular_vel_xy)
        aug_vec_ee_vel_xy = rot.dot(vec_ee_vel_xy)

        aug_vec = np.concatenate(
            [
                aug_vec_ee_xy,
                vec_ee_z,
                aug_vec_block_xy,
                vec_block_z,
                aug_vec_relative_diff_xy,
                vec_relative_diff_z,
                vec_joint_right_finger_disp,
                vec_joint_left_finger_disp,
                vec_obj_orientation_euler_xy,
                aug_obj_orientation_euler_cos_sin_z,
                aug_vec_relative_linear_vel_xy,
                vec_relative_linear_vel_z,
                aug_vec_angular_vel_xy,
                vec_angular_vel_z,
                aug_vec_ee_vel_xy,
                vec_ee_vel_z,
                vec_joint_right_finger_vel,
                vec_joint_left_finger_vel,
            ]
        )

        vec_dict[key] = aug_vec

    aug_obs = vec_dict["obs"]
    aug_next_obs = vec_dict["next_obs"]
    aug_goal = vec_dict["goal"]
    aug_random_goal = vec_dict["random_goal"]

    return aug_obs, aug_next_obs, aug_action, aug_goal, aug_random_goal


def perturbVec_fetch_push_pick2(obs, next_obs, action, goal, random_goal):
    aug_obs = obs.copy()
    aug_next_obs = next_obs.copy()
    aug_action = action.copy()
    aug_goal = goal.copy()
    aug_random_goal = random_goal.copy()

    theta = np.random.random() * 2 * np.pi - np.pi
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotate the first 2 elements of the action i.e. x,y of the end effector
    aug_action[0:2] = rot.dot(aug_action[0:2])
    # Ensure the action is within the limits of -1 and 1
    aug_action[0] = np.clip(aug_action[0], -1, 1)
    aug_action[1] = np.clip(aug_action[1], -1, 1)

    # Rotate the Observation, Next Observation, Goal, Random Goal
    vec_dict = {
        "obs": obs,
        "next_obs": next_obs,
        "goal": goal,
        "random_goal": random_goal,
    }

    # print("Aug Obs: ", aug_obs)
    # print("Aug Next Obs: ", aug_next_obs)
    # print("Aug Action: ", aug_action)
    # print("Aug Goal: ", aug_goal)
    # print("Aug Random Goal: ", aug_random_goal)

    # Rotate the Observation, Next Observation, Goal, Random Goal
    for key, vec in vec_dict.items():
        vec_ee_xy = vec[0:2]
        vec_ee_z = vec[2:3]

        vec_block_xy = vec[3:5]
        vec_block_z = vec[5:6]

        vec_relative_diff_xy = vec[6:8]
        vec_relative_diff_z = vec[8:9]

        vec_joint_right_finger_disp = vec[9:10]
        vec_joint_left_finger_disp = vec[10:11]

        vec_obj_orientation_euler_xy = vec[11:13]
        vec_obj_orientation_euler_cos_sin_z = vec[13:15]

        vec_relative_linear_vel_xy = vec[15:17]
        vec_relative_linear_vel_z = vec[17:18]

        vec_angular_vel_xy = vec[18:20]
        vec_angular_vel_z = vec[20:21]

        vec_ee_vel_xy = vec[21:23]
        vec_ee_vel_z = vec[23:24]

        vec_joint_right_finger_vel = vec[24:25]
        vec_joint_left_finger_vel = vec[25:26]

        aug_vec_ee_xy = rot.dot(vec_ee_xy)
        aug_vec_block_xy = rot.dot(vec_block_xy)
        aug_vec_relative_diff_xy = rot.dot(vec_relative_diff_xy)

        # aug_obj_orientation_euler_cos_sin_z = rot.dot(obj_orientation_euler_cos_sin_z)

        aug_vec_relative_linear_vel_xy = rot.dot(vec_relative_linear_vel_xy)
        aug_vec_angular_vel_xy = rot.dot(vec_angular_vel_xy)
        aug_vec_ee_vel_xy = rot.dot(vec_ee_vel_xy)

        current_theta = extract_angles_from_batch_numpy(
            vec_obj_orientation_euler_cos_sin_z[0],
            vec_obj_orientation_euler_cos_sin_z[1],
        )
        aug_current_theta = current_theta + theta
        if aug_current_theta > np.pi:
            aug_current_theta -= 2 * np.pi
        if aug_current_theta < -np.pi:
            aug_current_theta += 2 * np.pi
        aug_obj_orientation_euler_cos_sin_z = np.array(
            [np.cos(aug_current_theta), np.sin(aug_current_theta)]
        )

        aug_vec = np.concatenate(
            [
                aug_vec_ee_xy,
                vec_ee_z,
                aug_vec_block_xy,
                vec_block_z,
                aug_vec_relative_diff_xy,
                vec_relative_diff_z,
                vec_joint_right_finger_disp,
                vec_joint_left_finger_disp,
                vec_obj_orientation_euler_xy,
                aug_obj_orientation_euler_cos_sin_z,
                aug_vec_relative_linear_vel_xy,
                vec_relative_linear_vel_z,
                aug_vec_angular_vel_xy,
                vec_angular_vel_z,
                aug_vec_ee_vel_xy,
                vec_ee_vel_z,
                vec_joint_right_finger_vel,
                vec_joint_left_finger_vel,
            ]
        )

        vec_dict[key] = aug_vec

    aug_obs = vec_dict["obs"]
    aug_next_obs = vec_dict["next_obs"]
    aug_goal = vec_dict["goal"]
    aug_random_goal = vec_dict["random_goal"]

    return aug_obs, aug_next_obs, aug_action, aug_goal, aug_random_goal


def extract_angles_from_batch_numpy(cos_X_batch, sin_X_batch):
    """
    Extract angles from batch outputs of cos X and sin X using NumPy.

    :param cos_X_batch: NumPy array of cos X values
    :param sin_X_batch: NumPy array of sin X values
    :return: Tuple of arrays (X_batch in radians, X_degrees_batch in degrees)
    """
    # Ensure inputs are NumPy arrays
    cos_X_batch = np.asarray(cos_X_batch, dtype=np.float64)
    sin_X_batch = np.asarray(sin_X_batch, dtype=np.float64)

    # Compute the magnitude for normalization
    r_batch = np.hypot(cos_X_batch, sin_X_batch)  # Shape: [batch_size, ...]

    # Avoid division by zero
    epsilon = 1e-8
    r_batch_safe = np.where(r_batch == 0, epsilon, r_batch)

    # Normalize
    cos_X_norm = cos_X_batch / r_batch_safe
    sin_X_norm = sin_X_batch / r_batch_safe

    # Compute the angles in radians
    X_batch = np.arctan2(sin_X_norm, cos_X_norm)  # Shape: [batch_size, ...]

    # Adjust angle range to [0, 2π)
    X_batch = np.where(X_batch < 0, X_batch + 2 * np.pi, X_batch)

    return X_batch


def augmentBatch_SO2_fetch_push_pick(batch):
    obs, next_obs, action, goal, random_goal = (
        batch["obs"],
        batch["obs_next"],
        batch["actions"],
        batch["g"],
        batch["random_g"],
    )
    aug_obs, aug_next_obs, aug_action, aug_goal, aug_random_goal = (
        np.zeros_like(obs),
        np.zeros_like(next_obs),
        np.zeros_like(action),
        np.zeros_like(goal),
        np.zeros_like(random_goal),
    )

    # obs: (batch, ... )

    # print(aug_obs[23,:])
    for i in range(obs.shape[0]):
        # aug_obs[i], aug_next_obs[i], aug_action[i], aug_goal[i], aug_random_goal[i] = perturbVec_fetch_push_pick(obs[i], next_obs[i], action[i], goal[i], random_goal[i])
        (
            aug_obs[i, :],
            aug_next_obs[i, :],
            aug_action[i, :],
            aug_goal[i, :],
            aug_random_goal[i, :],
        ) = perturbVec_fetch_push_pick2(
            obs[i, :], next_obs[i, :], action[i, :], goal[i, :], random_goal[i, :]
        )
    # print(aug_obs[23,:])
    # asdasdasd

    augmented_batch = {}
    augmented_batch["obs"] = aug_obs
    augmented_batch["obs_next"] = aug_next_obs
    augmented_batch["actions"] = aug_action
    augmented_batch["g"] = aug_goal
    augmented_batch["random_g"] = aug_random_goal

    return augmented_batch
