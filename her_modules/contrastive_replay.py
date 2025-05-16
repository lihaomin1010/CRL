import numpy as np


class contrastive_sampler:
    def __init__(self, replay_strategy, replay_k, gamma, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func
        self.gamma = gamma

    def sample_her_transitions(
        self, episode_batch, batch_size_in_transitions, train=False
    ):
        T = episode_batch["actions"].shape[1]
        rollout_batch_size = episode_batch["actions"].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        # episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)\

        # Sample batch_size random unique integers using numpy with low as 0 and high as rollout_batch_size
        if train:
            episode_idxs = np.random.choice(
                rollout_batch_size, batch_size, replace=False
            )
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)

        t_samples = np.random.randint(T, size=batch_size)
        transitions = {
            key: episode_batch[key][episode_idxs, t_samples].copy()
            for key in episode_batch.keys()
        }

        # positive idx
        add = np.random.geometric(p=(1 - self.gamma), size=batch_size)
        # future_offset = np.random.uniform(size=batch_size) * add    #(3) #(T - t_samples) #(np.random.geometric(p=self.gamma, size=1)[0])
        future_offset = add
        # future_offset = future_offset.astype(int)

        future_t = t_samples + 1 + future_offset
        future_t[future_t >= T] = T - 1

        # Random Goals for Actor
        episode_idxs_2 = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples_2 = np.random.randint(T, size=batch_size)
        transitions["random_g"] = episode_batch["ag"][
            episode_idxs_2, t_samples_2
        ].copy()

        # replace go with achieved goal
        future_ag = episode_batch["ag"][episode_idxs, future_t]
        transitions["g"] = future_ag


        mid_offest = np.random.randint(0, T, batch_size)
        mid_t = np.minimum(mid_offest, future_t)
        transitions["mid_g"] = episode_batch["ag"][episode_idxs, mid_t].copy()

        # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # to get the params to re-compute reward
        # transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {
            k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
            for k in transitions.keys()
        }

        return transitions
