import threading
import numpy as np
from random import sample

"""
the replay buffer here is basically from the openai baselines code
"""

def sample_transitions(episode_batch, batch_size_in_transitions=None,doneAlive=False):

    if doneAlive:
        capacity = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which time steps to be used
        t_samples = np.random.randint(0, capacity, size=batch_size)
        transitions = {key: episode_batch[key][t_samples] for key in episode_batch.keys()}
    else:
        rollout_batch_size = episode_batch['actions'].shape[0]
        if batch_size_in_transitions is None:
            transitions = {k: episode_batch[k].reshape(-1,*np.array(episode_batch[k]).shape[2:]) for k in episode_batch.keys()}
        else:
            batch_size = batch_size_in_transitions
            # select which rollouts and which timesteps to be used
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            T = episode_batch['actions'].shape[1]
            t_samples = np.random.randint(T, size=batch_size)
            transitions = {key: episode_batch[key][episode_idxs, t_samples] for key in episode_batch.keys()}

    return transitions


class BaseReplayBuffer():
    def __init__(self, config,env_params):
        self.env_params = env_params
        self.doneAlive = config['doneAlive']
        self.full = False
        if self.doneAlive:
            self.idx = 0
            self.capacity = config['bufferCapacity']
        else:
            self.horizon = int(config['max_episode_steps'] // config['actionRepeat'])
            self.numEpBuffer = config['numEpBuffer']
            self.capacity = self.numEpBuffer * self.horizon
            self.all_idx = list(range(self.numEpBuffer))
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.method = config['method']


    def update_current_size(self, inc):
        "only if doneAlive"
        self.current_size = min(self.capacity, self.current_size + inc)
        if self.current_size == self.capacity and not self.full:
            self.full = True

    def get_storage_idx(self, inc=None):
        "only if not doneAlive"
        inc = inc or 1
        if self.current_size + inc <= self.numEpBuffer:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.numEpBuffer:
            overflow = inc - (self.numEpBuffer - self.current_size)
            idx_a = np.arange(self.current_size, self.numEpBuffer)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = sample(self.all_idx, inc)

        if not self.full:
            self.current_size = min(self.numEpBuffer, self.current_size + inc)
            if self.current_size == self.numEpBuffer:
                self.full = True
        return idx


class Replay_buffer(BaseReplayBuffer):
    def __init__(self, config,env_params):
        BaseReplayBuffer.__init__(self, config,env_params)
        # create the buffer to store info
        if self.doneAlive:
            self.buffers = {'obs': np.empty([self.capacity, *[i for i in self.env_params['obs']]], dtype=np.float32),
                            'obs_next': np.empty([self.capacity, *[i for i in self.env_params['obs']]], dtype=np.float32),
                            'r': np.empty([self.capacity, 1], dtype=np.float32),
                            'not_dones': np.empty([self.capacity, 1], dtype=np.float32),
                            'actions': np.empty([self.capacity, self.env_params['action']], dtype=np.float32),
                            }
        else:
            self.buffers = {'obs': np.empty([self.numEpBuffer, self.horizon + 1, *[i for i in self.env_params['obs']]], dtype=np.float32),  # *self.env_params['obs']]),
                            'r': np.empty([self.numEpBuffer, self.horizon, 1], dtype=np.float32),
                            'actions': np.empty([self.numEpBuffer, self.horizon, self.env_params['action']], dtype=np.float32),
                            }
        # thread lock
        self.lock = threading.Lock()


    def add(self, obs_step):
        "only if doneAlive"
        obs, obs_next, r, done, a = obs_step
        if not self.full:
            self.update_current_size(1)
        # with self.lock:
        np.copyto(self.buffers['obs'][self.idx], obs)
        np.copyto(self.buffers['obs_next'][self.idx], obs_next)
        np.copyto(self.buffers['r'][self.idx], r)
        np.copyto(self.buffers['not_dones'][self.idx],not done)
        np.copyto(self.buffers['actions'][self.idx], a)

        self.idx = (self.idx + 1) % self.capacity


    # store the episode
    def store_episode(self, episode_batch):
        "only if not doneAlive"
        mb_obs, mb_r, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            # store the informations
            self.idxs = self.get_storage_idx(inc=batch_size)
            self.buffers['obs'][self.idxs] = mb_obs
            self.buffers['r'][self.idxs] = mb_r
            self.buffers['actions'][self.idxs] = mb_actions
            self.n_transitions_stored += batch_size

    def sample(self, batch_size):
        # sample the data from the replay buffer
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] =  self.buffers[key][:self.current_size]
        if not self.doneAlive:
            temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        # sample transitions
        transitions = sample_transitions(temp_buffers, batch_size,doneAlive=self.doneAlive)
        return transitions
