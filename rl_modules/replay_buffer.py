import threading
import numpy as np
import pickle

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params["max_timesteps"]
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {
            "obs": np.empty([self.size, self.T + 1, self.env_params["obs"]]),
            "ag": np.empty([self.size, self.T + 1, self.env_params["goal"]]),
            "g": np.empty([self.size, self.T, self.env_params["goal"]]),
            "actions": np.empty([self.size, self.T, self.env_params["action"]]),
        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers["obs"][idxs] = mb_obs
            self.buffers["ag"][idxs] = mb_ag
            self.buffers["g"][idxs] = mb_g
            self.buffers["actions"][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size, train=False):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][: self.current_size]
        temp_buffers["obs_next"] = temp_buffers["obs"][:, 1:, :]
        temp_buffers["ag_next"] = temp_buffers["ag"][:, 1:, :]
        # sample transitions
        # transitions = self.sample_func(temp_buffers, batch_size,train=train)
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
    

    def save_replay_buffer(self, file_path):
        """
        Save the replay buffer's state to a file.

        Args:
            file_path (str): The path (including filename) where the replay buffer state will be saved.
        """
        # Prepare a dictionary with the state to save.
        state = {
            'env_params': self.env_params,
            'T': self.T,
            'size': self.size,
            'current_size': self.current_size,
            'n_transitions_stored': self.n_transitions_stored,
            'buffers': self.buffers
        }

        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Replay buffer saved to {file_path}")




    def load_replay_buffer(self, file_path):
        """
        Load the replay buffer's state from a file.

        Args:
            file_path (str): The path to the file containing the replay buffer state.
        """
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        self.env_params = state['env_params']
        self.T = state['T']
        self.size = state['size']
        self.current_size = state['current_size']
        self.n_transitions_stored = state['n_transitions_stored']
        self.buffers = state['buffers']
        
        # Recreate the lock as it isn't serialisable.
        self.lock = threading.Lock()
        
        print(f"Replay buffer loaded from {file_path}")


class replay_buffer_img:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params["max_timesteps"]
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        check = np.zeros((self.size, self.T, self.env_params["obs"]), dtype="uint8")
        print(check.shape)
        self.buffers = {
            "obs": np.zeros(
                (self.size, self.T + 1, self.env_params["obs"]), dtype="uint8"
            ),
            "ag": np.zeros(
                (self.size, self.T + 1, self.env_params["goal"]), dtype="uint8"
            ),
            "g": np.zeros((self.size, self.T, self.env_params["goal"]), dtype="uint8"),
            "actions": np.empty([self.size, self.T, self.env_params["action"]]),
        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers["obs"][idxs] = mb_obs
            self.buffers["ag"][idxs] = mb_ag
            self.buffers["g"][idxs] = mb_g
            self.buffers["actions"][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size, train=False):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][: self.current_size]
        temp_buffers["obs_next"] = temp_buffers["obs"][:, 1:, :]
        temp_buffers["ag_next"] = temp_buffers["ag"][:, 1:, :]
        # sample transitions
        # transitions = self.sample_func(temp_buffers, batch_size,train=train)
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
