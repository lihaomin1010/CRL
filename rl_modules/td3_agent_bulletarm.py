import torch
import os
from datetime import datetime
import numpy as np


from rl_modules.replay_buffer import replay_buffer
from rl_modules.td3_models import deterministic_actor, flatten_mlp
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

import wandb


"""
TD3 with HER

"""
class td3_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.timesteps = 0
        self.target_updates = 0
        self.grad_updates = 0
        

        self.p_range = env_params['action_ranges'][:,0]
        self.dx_range =  env_params['action_ranges'][:,1]
        self.dy_range =  env_params['action_ranges'][:,2]
        self.dz_range = env_params['action_ranges'][:,3]
        self.dtheta_range = env_params['action_ranges'][:,4]

        # create the network
        self.actor_network = deterministic_actor(env_params['obs'] + env_params['goal'], env_params['action'], 256)
        self.critic_network1 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        self.critic_network2 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        
        
        # build up the target network
        self.actor_target_network = deterministic_actor(env_params['obs'] + env_params['goal'], env_params['action'], 256)
        self.critic_target_network1 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        self.critic_target_network2 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network1.cuda()
            self.critic_network2.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network1.cuda()
            self.critic_target_network2.cuda()
        
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)
        self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, None)
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        self.expert_buffer = replay_buffer(
            self.env_params,
            self.args.buffer_size,
            self.her_module.sample_her_transitions,
        )
        
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)


        wandb.login()
        
        config={
                "learning_rate": args.lr_actor,
                "algorithm": "TD3 + HER",
                "env": args.env_name,
                "seed": args.seed,
                "Task": "Contrastive RL",
            }
        config.update(args.__dict__)

        run = wandb.init(
            project="Equi_Contrastive_RL",
            config = config,
            mode = "disabled" if args.disable_wandb else "online",
        )
        
    
    def process_obs(self, obs):
        """
        process the observation

        """
        obs_dict = {}
        obs = obs[2]
        
        obs_dict["observation"] = obs[:,: self.env_params["obs"]]
        obs_dict["desired_goal"] = obs[:,self.env_params["obs"] : 2 * self.env_params["obs"]]
        obs_dict["achieved_goal"] = obs[:, 2 * self.env_params["obs"] : 3 * self.env_params["obs"]]
        return obs_dict
    

    def decodeActions(self,unscaled_action):
        """
        Scale the action in range of (-1, 1) into the true scale
        :param args: unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        try:
            unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta = unscaled_action[0]
        except:
            unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz = unscaled_action[0]
        

        p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
        dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
        dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
        dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

        if unscaled_action.shape[1] == 5:
            dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
            actions = np.array([p, dx, dy, dz, dtheta]).reshape(1,5)
        else:
            actions = np.array([p, dx, dy, dz]).reshape(1,4)
        
        return actions
    


    def encodeActions(self, true_action):
        """
        Encode the action from the true scale into the normalized range (-1, 1).
        
        This is the inverse of decodeActions. For each action component, the normalized
        value is computed as:
        
            unscaled_value = 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1
            
        The function supports both 4-dimensional actions (p, dx, dy, dz) and 
        5-dimensional actions (p, dx, dy, dz, dtheta).
        
        :param true_action: np.array of shape (1, 4) or (1, 5) with actions in true scale.
        :return: normalized_actions: np.array of the same shape with actions in range (-1, 1).
        """
        try:
            # Try to unpack a 5-dimensional action
            p, dx, dy, dz, dtheta = true_action[0]
        except:
            # Otherwise, assume 4-dimensional
            p, dx, dy, dz = true_action[0]

        # Encode each action component from true scale to normalized scale.
        unscaled_p = 2 * (p - self.p_range[0]) / (self.p_range[1] - self.p_range[0]) - 1
        unscaled_dx = 2 * (dx - self.dx_range[0]) / (self.dx_range[1] - self.dx_range[0]) - 1
        unscaled_dy = 2 * (dy - self.dy_range[0]) / (self.dy_range[1] - self.dy_range[0]) - 1
        unscaled_dz = 2 * (dz - self.dz_range[0]) / (self.dz_range[1] - self.dz_range[0]) - 1

        if true_action.shape[1] == 5:
            unscaled_dtheta = 2 * (dtheta - self.dtheta_range[0]) / (self.dtheta_range[1] - self.dtheta_range[0]) - 1
            normalized_actions = np.array([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta]).reshape(1, 5)
        else:
            normalized_actions = np.array([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz]).reshape(1, 4)

        return normalized_actions

    
    def learn(self):
        """
        train the network

        """
        # Load the expert buffer if it exists
        if os.path.exists("expert_buffers"):
            self.expert_buffer.load_replay_buffer(
                os.path.join("expert_buffers", "expert_demos_{}_{}.pkl".format(self.args.env_name, self.args.num_bc_demos))
            )
            print("Loaded Expert Buffer")
        else:
            print("No Expert Buffer Found")

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    observation = self.process_obs(observation)
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
        
                    
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                            self.timesteps += 1
                        

                        if self.timesteps > self.args.init_exploration_steps and self.timesteps % self.args.n_updates == 0:
                            for _ in range(self.args.n_updates):
                                # train the network
                                self._update_network()
                            

                        # feed the actions into the environment
                        decoded_action = self.decodeActions(action)
                        observation_new, _, _ = self.env.step(decoded_action)


                        observation_new = self.process_obs(observation_new)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
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
                mb_obs = np.array(mb_obs).squeeze()
                mb_ag = np.array(mb_ag).squeeze()
                mb_g = np.array(mb_g).squeeze()
                mb_actions = np.array(mb_actions).squeeze()
        
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        
            
            # start to do the evaluation
            success_rate = self._eval_agent()
            print('[{}] epoch is: {}, eval success rate is: {:.3f}, Timesteps: {}, Grad_Updates: {}'.format(datetime.now(), epoch, success_rate,self.timesteps, self.grad_updates))

    
    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        # obs_norm = self.o_norm.normalize(obs)
        # g_norm = self.g_norm.normalize(g)
        obs_norm = obs
        g_norm = g



        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    
    
    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        
        # Unsqueeze the aciton numpy
        action = np.expand_dims(action, axis=0)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        
        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        
        # start to do the update
        # obs_norm = self.o_norm.normalize(transitions['obs'])
        # g_norm = self.g_norm.normalize(transitions['g'])
        obs_norm = transitions['obs']
        g_norm = transitions['g']

        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        # obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        # g_next_norm = self.g_norm.normalize(transitions['g_next'])
        obs_next_norm = transitions['obs_next']
        g_next_norm = transitions['g_next']
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        
        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            noise = (torch.randn_like(actions_next) * self.args.target_noise_eps * self.env_params['action_max']).clamp(-0.5, 0.5)
            actions_next = (actions_next + noise).clamp(-self.env_params['action_max'], self.env_params['action_max'])


            q_next_value1 = self.critic_target_network1(inputs_next_norm_tensor, actions_next)
            q_next_value2 = self.critic_target_network2(inputs_next_norm_tensor, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2)
            
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            
        
        # the q loss for critic network1
        real_q_value1 = self.critic_network1(inputs_norm_tensor, actions_tensor)
        critic_loss1 = (target_q_value - real_q_value1).pow(2).mean()
        
        # the q loss for critic network2
        real_q_value2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
        critic_loss2 = (target_q_value - real_q_value2).pow(2).mean()
        
        # update the critic_network1
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        # update the critic_network2
        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        self.grad_updates += 1
        if self.grad_updates % 2 == 0:
            # the actor loss
            actions_real = self.actor_network(inputs_norm_tensor)
            actor_loss = -self.critic_network1(inputs_norm_tensor, actions_real).mean()
            # actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

            #################################################################################
            if self.args.use_bc_with_rl:
                # --- Sample and process expert transitions for BC loss ---
                expert_transitions = self.expert_buffer.sample(self.args.batch_size, train=True)
                o_e, o_next_e, g_e, random_g_e = (
                    expert_transitions["obs"],
                    expert_transitions["obs_next"],
                    expert_transitions["g"],
                    expert_transitions["random_g"],
                )
                expert_transitions["obs"], expert_transitions["g"] = self._preproc_og(o_e, g_e)

                # obs_e_norm = self.o_norm.normalize(expert_transitions["obs"])
                # g_e_norm = self.g_norm.normalize(expert_transitions["g"])
                obs_e_norm = expert_transitions["obs"]
                g_e_norm = expert_transitions["g"]
                inputs_e_norm = np.concatenate([obs_e_norm, g_e_norm], axis=1)

                inputs_e_norm_tensor = torch.tensor(inputs_e_norm, dtype=torch.float32)
                actions_expert_tensor = torch.tensor(expert_transitions["actions"], dtype=torch.float32)

                if self.args.cuda:
                    inputs_e_norm_tensor = inputs_e_norm_tensor.cuda()
                    actions_expert_tensor = actions_expert_tensor.cuda()

                # Forward pass on expert transitions using actor network
                actions_expert_pred = self.actor_network(inputs_e_norm_tensor)

                # --- Compute BC loss (supervised MSE loss) ---
                bc_loss = torch.nn.functional.mse_loss(actions_expert_pred, actions_expert_tensor)

                # --- Combine RL actor loss and BC loss ---
                # self.args.lambda_bc is a hyperparameter to weight the BC loss
                actor_loss = actor_loss + 1 * bc_loss
            #################################################################################
            
            
            # start to update the network
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self._soft_update_target_network(self.actor_target_network, self.actor_network)
            self._soft_update_target_network(self.critic_target_network1, self.critic_network1)
            self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
        
    

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            observation = self.process_obs(observation)
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy()

                actions = actions.squeeze() 
                actions=np.expand_dims(actions, axis=0)
                decoded_action = self.decodeActions(actions)
                observation_new, reward, _  = self.env.step(decoded_action)
                observation_new = self.process_obs(observation_new)
                obs = observation_new['observation']
                g = observation_new['desired_goal']

                if reward == 1.0:
                    break
                
            total_success_rate.append(reward)
        total_success_rate = np.array(total_success_rate)
        

        local_success_rate = np.mean(total_success_rate)
        wandb.log({"Evaluated Reward": local_success_rate}, step=self.timesteps)
        return local_success_rate