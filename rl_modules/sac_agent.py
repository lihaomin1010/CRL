import torch
import os
from datetime import datetime
import numpy as np


from rl_modules.replay_buffer import replay_buffer
from rl_modules.sac_models import tanh_gaussian_actor, flatten_mlp
from rl_modules.utils import get_action_info
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from her_modules.rnd import RND
from her_modules.normalize import Normalizer
from her_modules.skill_entropy import CRL
import wandb


"""
SAC with HER

"""
class sac_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.timesteps = 0
        self.target_updates = 0
        self.grad_updates = 0
        self.action_max = env_params['action_max']
        
        # create the network
        self.actor_network = tanh_gaussian_actor(env_params['obs'] + env_params['goal'], env_params['action'], 256, self.args.log_std_min, self.args.log_std_max)
        self.critic1 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        self.critic2 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        
        
        # build up the target network
        self.actor_target_network = tanh_gaussian_actor(env_params['obs'] + env_params['goal'], env_params['action'], 256, self.args.log_std_min, self.args.log_std_max)
        self.critic_target_network1 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        self.critic_target_network2 = flatten_mlp(env_params['obs'] + env_params['goal'], 256, env_params['action'])
        
        # load the weights into the target networks
        self.critic_target_network1.load_state_dict(self.critic1.state_dict())
        self.critic_target_network2.load_state_dict(self.critic2.state_dict())

        # self.target_entropy = -1 * self.env.action_space.shape[0]
        self.target_entropy = -1 * env_params['action']
        # self.target_entropy = 0
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.log_beta = torch.zeros(1, requires_grad=True)

        self.entropy_temp = args.entropy_temp
        self.steps = 0
        self.rnd_worker = None
        if args.rnd:
            self.rnd_worker = RND(env_params['obs'] + env_params['action'], env_params['obs'], name="sac")

        self.crl_worker = None
        if args.crl:
            self.crl_worker = CRL(env_params['obs'] + env_params['action'], 4, buffer_size=10000, n_updates=4)

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic1.cuda()
            self.critic2.cuda()
            self.critic_target_network1.cuda()
            self.critic_target_network2.cuda()
            self.actor_target_network.cuda()
            self.log_alpha.cuda()
            self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
            self.log_beta.cuda()
            self.log_beta = torch.zeros(1, requires_grad=True, device='cuda')
        
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim1 = torch.optim.Adam(self.critic1.parameters(), lr=self.args.lr_critic)
        self.critic_optim2 = torch.optim.Adam(self.critic2.parameters(), lr=self.args.lr_critic)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_actor)
        self.beta_optim = torch.optim.Adam([self.log_beta], lr=self.args.lr_actor)

        # her sampler
        # self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k)
        
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        #wandb.login()
        
        config={
                "learning_rate": args.lr_actor,
                "algorithm": "SAC + HER",
                "env": args.env_name,
                "seed": args.seed,
                "Task": "Contrastive RL",
            }
        config.update(args.__dict__)

        # run = wandb.init(
        #     project="Equi_Contrastive_RL",
        #     config = config,
        #     mode = "disabled" if args.disable_wandb else "online",
        # )

    
    def process_obs(self, obs):
        o = dict()
        o['observation'] = obs[:self.env_params['obs']]
        o['desired_goal'] = obs[self.env_params['obs']:-self.env_params['goal']]
        o['achieved_goal'] = obs[-self.env_params['goal']:]

        return o


    def learn(self):

        self.visualize_trajectories("saved_models/evaluation_trajectories/test.npy")
        """
        train the network

        """
        t_begin = datetime.now().timestamp()
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
                            action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                            action = action.cpu().numpy()[0]
                            self.timesteps += 1
                        
                        if self.timesteps > self.args.init_exploration_steps and self.timesteps % self.args.n_updates == 0:
                            for _ in range(self.args.n_updates):
                                # train the network
                                self._update_network()
                                self.steps += 1
                                if (self.steps + 1) % self.args.record_interval == 0:
                                    wandb.log({"log_alpha": self.log_alpha}, step=self.steps+1)
                            self.grad_updates += self.args.n_updates
                            
                            # soft update
                            if self.target_updates % self.args.target_update_interval == 0:
                                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                            self._soft_update_target_network(self.critic_target_network1, self.critic1)
                            self._soft_update_target_network(self.critic_target_network2, self.critic2)
                            self.target_updates += 1

                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
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
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
        
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

            if (epoch+1) % self.args.save_interval == 0:
                self.save_checkpoint(f"{self.args.save_path}/{t_begin}/{epoch+1}")
                
            
            # start to do the evaluation
            success_rate = self._eval_agent()
            print('[{}] epoch is: {}, eval success rate is: {:.3f}, Timesteps: {}, Grad_Updates: {}'.format(datetime.now(), epoch, success_rate,self.timesteps, self.grad_updates))

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
        init_o = o.copy()
        init_g = g.copy()
        
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)

        # if (np.sum(init_o - o) != 0) and (np.sum(init_g - g) != 0):
        #     print('clipping')

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
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        rnd_inputs_norm = np.concatenate([obs_norm, transitions['actions']], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        # obs_next_norm = transitions['obs_next']
        # g_next_norm = transitions['g_next']
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        rnd_inputs_norm_tensor = torch.tensor(rnd_inputs_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        obs_next_norm_tensor = torch.tensor(obs_next_norm, dtype=torch.float32)

        if self.args.cuda:
            obs_norm_tensor = obs_norm_tensor.cuda()
            rnd_inputs_norm_tensor = rnd_inputs_norm_tensor.cuda()
            obs_next_norm_tensor = obs_next_norm_tensor.cuda()
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        if self.args.rnd:
            need_train = True
            if self.args.stop_rnd and self.steps > 3000:
                need_train = False
            _, intrinsic_reward_origin = self.rnd_worker.train(rnd_inputs_norm_tensor, obs_next_norm_tensor, need_train=need_train)
            #intrinsic_reward = self.rnd_worker.get_intrinsic_reward(inputs_next_norm_tensor, obs_next_norm_tensor)
            thre = torch.max(torch.abs(intrinsic_reward_origin))
            intrinsic_reward = intrinsic_reward_origin/thre
            r_tensor += self.args.rnd_num * intrinsic_reward

            skill_entropy = None
            if self.args.crl:
                #_ = self.crl_worker.train(rnd_inputs_norm_tensor, intrinsic_reward.detach())
                _ = self.crl_worker.train(rnd_inputs_norm_tensor, intrinsic_reward.detach())
                skill_entropy = self.crl_worker.get_skill_entropy(torch.concatenate((obs_norm_tensor, actions_tensor), dim=1))
                beta_loss = -(self.log_beta * ((self.target_entropy - skill_entropy).detach())).mean()
                self.beta_optim.zero_grad()
                beta_loss.backward()
                # sync_parameter(self.log_alpha)
                self.beta_optim.step()


        pis = self.actor_network(inputs_norm_tensor)
        actions_info = get_action_info(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)

        # use the automatically tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # sync_parameter(self.log_alpha)
        self.alpha_optim.step()

        alpha = self.log_alpha.exp()

        beta = self.log_beta.exp()

        # calculate the actor loss
        q_actions_ = torch.min(self.critic1(inputs_norm_tensor, actions_), self.critic2(inputs_norm_tensor, actions_))
        actor_loss = ( self.entropy_temp*alpha * log_prob - q_actions_).mean()
        if self.args.crl:
            actor_loss = actor_loss + self.entropy_temp * beta * skill_entropy.mean()
        # actor_loss = (- q_actions_).mean()

        #Calculate the critic loss
        q1_value = self.critic1(inputs_norm_tensor, actions_tensor)
        q2_value = self.critic2(inputs_norm_tensor, actions_tensor)
        
        with torch.no_grad():
            pis_next = self.actor_network(inputs_next_norm_tensor)
            actions_info_next = get_action_info(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)

            target_q_value_next = torch.min(self.critic_target_network1(inputs_next_norm_tensor, actions_next_), self.critic_target_network2(inputs_next_norm_tensor, actions_next_)) - self.entropy_temp* alpha * log_prob_next

            if self.args.crl:
                target_q_value_next = target_q_value_next - self.entropy_temp * beta * skill_entropy

            target_q_value = r_tensor + self.args.gamma * target_q_value_next
            #target_q_value = r_tensor + target_q_value_next

        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()

        
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # update the critic_network1
        self.critic_optim1.zero_grad()
        qf1_loss.backward()
        self.critic_optim1.step()

        # update the critic_network2
        self.critic_optim2.zero_grad()
        qf2_loss.backward()
        self.critic_optim2.step()

    
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
                    action = get_action_info(pi, cuda=self.args.cuda).select_actions(reparameterize=False,exploration=False)
                    action = action.cpu().numpy()[0]
                observation_new, reward, done, info = self.env.step(action)
                observation_new = self.process_obs(observation_new)
                obs = observation_new['observation']
                g = observation_new['desired_goal']

                if reward == 1.0:
                    break
                
            total_success_rate.append(reward)
        total_success_rate = np.array(total_success_rate)
        

        local_success_rate = np.mean(total_success_rate)


        wandb.log({"Evaluated Reward": local_success_rate}, step=self.steps)
        return local_success_rate

    def save_checkpoint(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        """
        保存模型检查点到指定路径
        
        Args:
            path (str): 保存检查点的路径
        """
        checkpoint = {
            'actor_network': self.actor_network.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target_network': self.actor_target_network.state_dict(),
            'critic_target_network1': self.critic_target_network1.state_dict(),
            'critic_target_network2': self.critic_target_network2.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim1': self.critic_optim1.state_dict(),
            'critic_optim2': self.critic_optim2.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'beta_optim': self.beta_optim.state_dict(),
            'log_alpha': self.log_alpha,
            'log_beta': self.log_beta,
        }
        
        if self.args.rnd and self.rnd_worker is not None:
            checkpoint['rnd_worker'] = self.rnd_worker.train_net.state_dict()
            checkpoint['rnd_worker_optim'] = self.rnd_worker.optimizer.state_dict()
        if self.args.crl and self.crl_worker is not None:
            checkpoint['crl_worker'] = self.crl_worker.train_net.state_dict()
            checkpoint['crl_worker_optim'] = self.crl_worker.optimizer.state_dict()

            
        torch.save(checkpoint, f"{path}/checkpoint.pth")
        print(f"检查点已保存到: {path}")

    def load_checkpoint(self, path):
        """
        从指定路径加载模型检查点
        
        Args:
            path (str): 加载检查点的路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到检查点文件: {path}")
            
        checkpoint = torch.load(path)
        
        # 加载网络参数
        self.actor_network.load_state_dict(checkpoint['actor_network'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target_network.load_state_dict(checkpoint['actor_target_network'])
        self.critic_target_network1.load_state_dict(checkpoint['critic_target_network1'])
        self.critic_target_network2.load_state_dict(checkpoint['critic_target_network2'])
        
        # 加载优化器状态
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic_optim1.load_state_dict(checkpoint['critic_optim1'])
        self.critic_optim2.load_state_dict(checkpoint['critic_optim2'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.beta_optim.load_state_dict(checkpoint['beta_optim'])
        
        # 加载其他参数
        self.log_alpha = checkpoint['log_alpha']
        self.log_beta = checkpoint['log_beta']
        
        # 加载RND和CRL（如果存在）
        if self.args.rnd and self.rnd_worker is not None and 'rnd_worker' in checkpoint:
            self.rnd_worker.train_net.load_state_dict(checkpoint['rnd_worker'])
            self.rnd_worker.optimizer.load_state_dict(checkpoint['rnd_worker_optim'])
            
        if self.args.crl and self.crl_worker is not None and 'crl_worker' in checkpoint:
            self.crl_worker.train_net.load_state_dict(checkpoint['crl_worker'])
            self.crl_worker.optimizer.load_state_dict(checkpoint['crl_worker_optim'])
            
        print(f"已从 {path} 加载检查点")

    def visualize_trajectories(self, trajectory_file):
        """
        读取轨迹数据并进行可视化
        Args:
            trajectory_file: 轨迹数据文件路径
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np

        # 读取轨迹数据
        trajectories = np.load(trajectory_file, allow_pickle=True)

        # 收集所有observations和actions
        all_obs_actions = []
        all_distances = []

        for traj in trajectories:
            obs = traj['observations']
            actions = traj['actions']
            achieved_goals = traj['achieved_goals']

            # 计算每个时间步的欧氏距离
            distances = np.linalg.norm(obs[:, 3:6] - achieved_goals[:, 3:6], axis=1)

            # 拼接observations和actions
            obs_actions = np.concatenate([obs, actions], axis=1)

            all_obs_actions.append(obs_actions)
            all_distances.append(distances)

        # 将所有数据合并
        all_obs_actions = np.concatenate(all_obs_actions, axis=0)
        all_distances = np.concatenate(all_distances, axis=0)

        all_obs_actions_tensor = torch.tensor(all_obs_actions).cuda()
        r_t = torch.rand(all_obs_actions_tensor.shape).cuda()
        input_tensor = torch.cat([all_obs_actions_tensor, r_t],dim=1)

        # 使用process_handle处理数据
        processed_data = self.crl_worker.train_net(input_tensor)

        # 使用t-SNE进行降维
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(processed_data.detach().cpu().numpy())

        # 创建散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                              c=all_distances,
                              cmap='viridis',
                              alpha=0.6)
        plt.colorbar(scatter, label='Distance between observation and achieved goal')
        plt.title('t-SNE visualization of trajectories')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')

        # 保存图像
        save_dir = os.path.join(self.args.save_dir, 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f'tsne_visualization_{timestamp}.png'))
        plt.close()

        print(f'[INFO] 可视化结果已保存到: {save_dir}')