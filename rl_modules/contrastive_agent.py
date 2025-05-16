import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads, sync_parameter
from rl_modules.replay_buffer import replay_buffer
from rl_modules.contrastive_models import tanh_gaussian_actor, flatten_mlp_contrastive
from rl_modules.utils import get_action_info
from mpi_utils.normalizer import normalizer
from her_modules.contrastive_replay import contrastive_sampler

from rl_modules.replay_buffer import replay_buffer
from rl_modules.contrastive_models import tanh_gaussian_actor, flatten_mlp_contrastive

from rl_modules.utils import augmentBatch_SO2_fetch_push_pick
import wandb

"""
State-Based Contrastive Learning for Goal-Conditioned Reinforcement Learning
"""


class contrastive_agent:
    def __init__(self, args, env, env_params):  # , rng):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.timesteps = 0
        self.target_updates = 0
        self.grad_updates = 0
        self.action_max = self.env.action_space.high
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.margin = 1e-8
        self.tri = args.tri
        self.epoch = 0
        self.need_print = True

        print("GPU:", self.args.cuda)
        if self.args.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.actor_network = tanh_gaussian_actor(
            env_params["obs"] + env_params["goal"],
            env_params["action"],
            256,
            self.args.log_std_min,
            self.args.log_std_max,
        ).to(self.device)
        self.critic1 = flatten_mlp_contrastive(
            obs_dims=env_params["obs"],
            goal_dims=env_params["goal"],
            hidden_size=256,
            repr_dims=64,
            action_dims=env_params["action"],
        ).to(self.device)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic1)

        self.target_entropy = -1 * self.env.action_space.shape[0]
        self.target_entropy = torch.tensor(self.target_entropy).to(self.device)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.args.lr_actor
        )
        self.critic_optim1 = torch.optim.Adam(
            self.critic1.parameters(), lr=self.args.lr_critic
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_actor)

        # contrastive sampler
        self.her_module = contrastive_sampler(
            self.args.replay_strategy,
            self.args.replay_k,
            self.args.gamma,
            self.env.compute_reward,
        )

        # create the replay buffer
        self.buffer = replay_buffer(
            self.env_params,
            self.args.buffer_size,
            self.her_module.sample_her_transitions,
        )

        # create the normalizer
        self.o_norm = normalizer(
            size=env_params["obs"], default_clip_range=self.args.clip_range
        )
        self.g_norm = normalizer(
            size=env_params["goal"], default_clip_range=self.args.clip_range
        )

    def process_observation(self, obs):
        """
        process the observation

        """
        obs_dict = {}
        obs_dict["observation"] = obs[: self.env_params["obs"]]
        obs_dict["desired_goal"] = obs[
                                   self.env_params["obs"]: 2 * self.env_params["obs"]
                                   ]
        obs_dict["achieved_goal"] = obs[
                                    2 * self.env_params["obs"]: 3 * self.env_params["obs"]
                                    ]
        return obs_dict

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []

                    # reset the environment
                    observation = self.env.reset()

                    observation = self.process_observation(observation)
                    obs = observation["observation"]
                    ag = observation["achieved_goal"]
                    g = observation["desired_goal"]

                    # start to collect samples
                    for t in range(self.env_params["max_timesteps"]):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = get_action_info(
                                pi, cuda=self.args.cuda
                            ).select_actions(reparameterize=False)
                            action = action.cpu().numpy()[0]
                            self.timesteps += 1

                        if (self.timesteps > self.args.init_exploration_steps) and (
                                self.timesteps % self.args.n_updates == 0
                        ):
                            for _ in range(self.args.n_updates):
                                self.grad_updates += 1

                                # train the network
                                self._update_network()

                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)

                        observation_new = self.process_observation(observation_new)
                        obs_new = observation_new["observation"]
                        ag_new = observation_new["achieved_goal"]

                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())

                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)

                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

            # start to do the evaluation
            success_rate = self._eval_agent()
            print(
                "[{}] epoch is: {}, eval success rate is: {:.3f}, Timesteps {}, Grad_Updates {}".format(
                    datetime.now(),
                    epoch,
                    success_rate,
                    self.timesteps,
                    self.grad_updates,
                )
            )
            self.epoch = epoch

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # obs_norm = obs
        # g_norm = g

        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {
            "obs": mb_obs,
            "ag": mb_ag,
            "g": mb_g,
            "actions": mb_actions,
            "obs_next": mb_obs_next,
            "ag_next": mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(
            buffer_temp, num_transitions
        )
        obs, mid_g, g = transitions["obs"], transitions["mid_g"], transitions["g"]
        # pre process the obs and g
        transitions["obs"], transitions["g"] = self._preproc_og(obs, g)
        transitions["obs"], transitions["mid_g"] = self._preproc_og(obs, mid_g)
        # update
        self.o_norm.update(transitions["obs"])
        self.g_norm.update(transitions["g"])
        self.g_norm.update(transitions["mid_g"])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.g_norm.update(transitions["mid_g"])

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size, train=True)

        # pre-process the observation and goal
        o, o_next, g, random_g, mg = (
            transitions["obs"],
            transitions["obs_next"],
            transitions["g"],
            transitions["random_g"],
            transitions["mid_g"],
        )
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)
        transitions["obs"], transitions["random_g"] = self._preproc_og(o, random_g)
        transitions["obs"], transitions["mid_g"] = self._preproc_og(o, mg)

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        mg_norm = self.g_norm.normalize(transitions["mid_g"])

        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        random_g_norm = self.g_norm.normalize(transitions["random_g"])

        inputs_actor_norm = np.concatenate([obs_norm, random_g_norm], axis=1)

        # transfer them into the tensor
        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
        mg_norm_tensor = torch.tensor(mg_norm, dtype=torch.float32)
        random_g_norm_tensor = torch.tensor(random_g_norm, dtype=torch.float32)
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_actor_norm_tensor = torch.tensor(inputs_actor_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32)

        if self.args.cuda:
            obs_norm_tensor = obs_norm_tensor.cuda()
            g_norm_tensor = g_norm_tensor.cuda()
            mg_norm_tensor = mg_norm_tensor.cuda()
            random_g_norm_tensor = random_g_norm_tensor.cuda()
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_actor_norm_tensor = inputs_actor_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()

        pis = self.actor_network(inputs_actor_norm_tensor)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)

        # use the automatically tuning
        alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        alpha = self.log_alpha.exp()

        # calculate the actor loss
        q_actions_ = self.calculate_q_values(
            obs_norm_tensor, random_g_norm_tensor, actions_
        )
        actor_loss = -q_actions_.mean()
        self.avg_q_values = q_actions_.mean().detach().cpu().item()

        # Calculate the critic loss
        qf1_loss = self.critic_loss(obs_norm_tensor, g_norm_tensor, actions_tensor, mg_norm_tensor)

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the critic_network1
        self.critic_optim1.zero_grad()
        qf1_loss.backward()
        self.critic_optim1.step()

    def calculate_q_values(
            self, observations_tensor, goals_tensor, actor_update_actions
    ):
        sa_repr, g_repr = self.critic1(
            observation=observations_tensor,
            goal=goals_tensor,
            action=actor_update_actions,
        )

        # Contrastive Learning with inner product
        q_values = torch.einsum("ik,ik->i", sa_repr, g_repr)

        return q_values

    def critic_loss(self, observations_tensor, goals_tensor, actions, mid_goal_tensor=None):
        sa_repr, g_repr = self.critic1(
            observation=observations_tensor, goal=goals_tensor, action=actions
        )

        sa_repr, mg_repr = self.critic1(
            observation=observations_tensor, goal=mid_goal_tensor, action=actions
        )

        # Contrastive Learning with inner product
        logits = torch.einsum("ik,jk->ij", sa_repr, g_repr)

        targets = torch.eye(self.args.batch_size, device=logits.device)

        dlog = torch.einsum("ik,ik->i", sa_repr, g_repr)
        mdlog = torch.einsum("ik,ik->i", sa_repr, mg_repr)

        if self.epoch % 5 == 0:
            if self.need_print:
                print("dlog=", dlog)
                print("mdlog=", mdlog)
                print(dlog - mdlog)
                self.need_print = False
        else:
            if not self.need_print:
                self.need_print = True
        if self.tri:
            loss = self.BCE_loss(logits, targets) + 0.1 * F.relu(dlog - mdlog + self.margin).mean()
        else:
            loss = self.BCE_loss(logits, targets)
        return loss

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []

            # reset the environment
            observation = self.env.reset()

            observation = self.process_observation(observation)
            obs = observation["observation"]
            g = observation["desired_goal"]

            for _ in range(self.env_params["max_timesteps"]):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(
                        reparameterize=False, exploration=False
                    )
                    action = action.cpu().numpy()[0]
                observation_new, reward, _, info = self.env.step(action)

                observation_new = self.process_observation(observation_new)
                obs = observation_new["observation"]
                g = observation_new["desired_goal"]

                per_success_rate.append(bool(reward))

                if reward == 1.0:
                    break

            total_success_rate.append(reward)
        total_success_rate = np.array(total_success_rate)
        success_rate = np.mean(total_success_rate)

        return success_rate
