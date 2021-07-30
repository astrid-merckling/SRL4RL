import torch
import numpy as np
from mpi4py import MPI

from bullet_envs.utils import env_with_goals

from SRL4RL.rl.utils.replay_buffer import Replay_buffer
from SRL4RL.rl.utils.normalizer import Normalizer, TimeNormalizer
from SRL4RL.rl.modules.agent_utils import Saver


class Agent(Saver):
    def __init__(self, config, env, env_params, runner):
        self.config = config
        self.minEpBuffer = config['minEpBuffer']
        self.hashCode = config['hashCode']
        self.env_name = config['env_name']
        self.with_progress = False
        if self.env_name in ['AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0',
                             'Walker2DBulletEnv-v0']:
            self.with_progress = True
        self.gamma = config['gamma']
        self.with_goal = config['with_goal']
        self.env = env
        self.env_params = env_params
        self.method = config['method']
        self.agent = config['agent']
        self.max_episode_steps = config['max_episode_steps']
        self.actionRepeat = config['actionRepeat']
        self.batch_size = config['batch_size']
        self.gradient_steps = config['gradient_steps']
        self.n_eval_rollouts = config['n_eval_rollouts']
        self.n_episodes_rollout = config['n_episodes_rollout']
        self.n_epochs = config['n_epochs']
        self.action_dim = env_params['action']
        self.eval_interval = config['eval_interval']
        self.actor_update_interval = config['actor_update_interval']
        self.target_update_interval = config['target_update_interval']
        self.polyak = config['polyak']
        self.cpu = torch.device('cpu')
        self.device = config['device']
        self.save_dir = config['save_dir']
        self.clip_obs = config['clip_obs']

        # create the replay buffer
        self.buffer = Replay_buffer(config, config['env_params'])

        self.maxStepsReal = int(config['max_episode_steps'] // config['actionRepeat'])
        # create the normalizer
        if config['dir']:
            self.o_norm = Normalizer(size=env_params['obs'][0], default_clip_range=config['clip_range'],
                                     initNormalizer=config, name='o')
            self.g_norm = Normalizer(size=env_params['goal'], default_clip_range=config['clip_range'], initNormalizer=config,
                                     name='g')
            del config['o_mean'], config['g_mean'], config['o_std'], config['g_std']
        else:
            if self.method == 'openLoop':
                self.o_norm = TimeNormalizer(self.maxStepsReal)
            else:
                self.o_norm = Normalizer(size=env_params['obs'][0], default_clip_range=config['clip_range'])
            self.g_norm = Normalizer(size=env_params['goal'], default_clip_range=config['clip_range'])

        self.automatic_entropy_tuning = None
        self.runner = runner
        self.RL_training = False
        self.doneAlive = config['doneAlive']
        if not self.doneAlive:
            self.config['buffer_size'] = self.buffer.numEpBuffer * self.buffer.horizon
            self.config['horizon'] = self.buffer.horizon
            assert self.n_episodes_rollout <= self.buffer.numEpBuffer, 'n_episodes_rollout too high'

        self.currentEpisodesRollout = self.n_episodes_rollout
        self.enoughRollout = False

        print(
            'maxStepsReal: {} \nenoughRollout: {} \ncurrentEpisodesRollout: {}'.format(
                self.maxStepsReal, self.enoughRollout,
                self.currentEpisodesRollout))
        # Init Saver class
        super().__init__(config, env)

    # do the evaluation
    def _eval_agent(self):
        print('eval_agent ......')
        g = None
        success_rates = np.zeros((self.n_eval_rollouts, self.maxStepsReal))
        if self.with_progress:
            sum_success_rates = np.zeros((self.n_eval_rollouts))
        for nep in range(self.n_eval_rollouts):
            observation = self.env.reset()
            if self.with_goal:
                state = observation['observation']
                g = observation['desired_goal']
            else:
                state = observation

            for step in range(self.maxStepsReal):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(state, g)
                    pi = self.runner.forward(self.actor_network, input_tensor, eval=True)
                observation_new, reward, done, info = self.env.step(pi)
                if self.env_name in env_with_goals:
                    if self.with_goal:
                        state_new = observation_new['observation']
                        g = observation_new['desired_goal']
                    else:
                        state_new = observation_new
                        info['is_success'] = reward + 1

                    if (info['is_success'] == 1.) or (step == self.maxStepsReal):
                        success_rates[nep, step] = info['is_success']
                        break
                else:
                    state_new = observation_new
                    if not self.with_progress:
                        success_rates[nep, step] = reward

                state = state_new
            if self.with_progress:
                sum_success_rates[nep] = self.env.rewardProgress

        if not self.with_progress:
            sum_success_rates = np.sum(success_rates, axis=1)
        mean_success_rates = np.mean(sum_success_rates)
        global_success_rate = MPI.COMM_WORLD.allreduce(mean_success_rates, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()

    # soft update
    def soft_update_target_network(self, target, source):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_((1 - self.polyak) * target_param.data + self.polyak * param.data)

    def learn(self):
        """
        train the network
        """

        g, ag = None, None
        srl_update = 0
        while self.elapsed_epochs < self.n_epochs:
            if not self.doneAlive:
                " self.currentEpisodesRollout vary with respect to the SRL training "
                mb_obs = np.zeros(
                    [self.currentEpisodesRollout, self.buffer.horizon + 1, *[i for i in self.env_params['obs']]])
                mb_r = np.zeros([self.currentEpisodesRollout, self.buffer.horizon, 1])
                mb_actions = np.zeros(
                    [self.currentEpisodesRollout, self.buffer.horizon, self.env_params['action']])
            if self.RL_training: self.elapsed_soft_update += 1
            Nroll = 0
            if self.doneAlive:
                past_idx = self.buffer.idx
                new_step_buffer = 0
            for _ in range(self.currentEpisodesRollout):
                if self.RL_training: self.elapsed_rollouts += 1
                n_steps = 0
                observation = self.env.reset()
                done = False
                if self.with_goal:
                    state = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                else:
                    state = observation

                "start new episode"
                while not done:
                    if self.RL_training or (not self.config['random_buffer']):
                        self.elapsed_steps += 1
                        random = False
                        input_tensor = self._preproc_inputs(state, g)
                    else:
                        random = True
                        input_tensor = None
                    with torch.no_grad():
                        "As kostrikov2020DrQ, sample randomly for data collection"
                        pi = self.runner.forward(self.actor_network, input_tensor, random=random, env=self.env)
                    # feed the actions into the environment
                    observation_new, reward, done, info = self.env.step(pi)
                    if self.with_goal:
                        state_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                    else:
                        state_new = observation_new

                    "add step to buffer in case of doneAlive"
                    if self.doneAlive:
                        obs_step = [state, state_new, reward, float(done), pi]
                        self.buffer.add(obs_step)
                        new_step_buffer += 1
                    else:
                        "update episodes"
                        mb_obs[Nroll, n_steps] = state
                        mb_r[Nroll, n_steps] = reward
                        mb_actions[Nroll, n_steps] = pi

                    "update state"
                    state = state_new
                    if self.with_goal: ag = ag_new

                    n_steps += 1
                    assert n_steps <= self.maxStepsReal, "wrong max_episode_steps"

                "add last state"
                if not self.doneAlive:
                    assert n_steps == self.maxStepsReal, 'not doneAlive and n_steps != self.maxStepsReal'
                    mb_obs[Nroll, n_steps] = state

                Nroll += 1

            if self.doneAlive:
                "Update normalizer before RL_training"
                maxStep = min(new_step_buffer + past_idx, self.buffer.capacity)
                self.o_norm.update_normalizer(self.buffer.buffers['obs'][past_idx:maxStep])
            else:
                episode_batch = [mb_obs, mb_r, mb_actions]
                self.buffer.store_episode(episode_batch)
                "Update normalizer before RL_training"
                self._update_normalizer(episode_batch)

            if self.RL_training:
                "train the RL network"
                self.train(training=True)
                for gradient_step in range(self.gradient_steps):
                    self.update_network(bool(gradient_step % self.actor_update_interval == 0))
                    self.elapsed_gradients += 1

                    "Update target networks"
                    if (gradient_step + 1) % self.target_update_interval == 0:
                        self.soft_update_target_network(self.critic_target_network, self.critic_network)
                self.train(training=False)

            if self.RL_training:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    "do the evaluation"
                    if self.elapsed_epochs % self.eval_interval == 0:
                        "Compute episode returns averaged over n_eval_rollouts"
                        success_rate = self._eval_agent()
                        "Update logs"
                        self.update_success(success_rate)
                        "Update early_stopper"
                        self.early_stopper(self.eval_patience, Nepochs=self.elapsed_epochs)
                        prefix = 'best' if self.early_stopper.counter == 0 else 'last'
                        "Save models"
                        self.save_chekpoint(prefix)
                        if (self.elapsed_epochs == self.min_Nepochs) or (success_rate >= self.min_reward):
                            if self.early_stopper.patience != self.newPatience:
                                self.early_stopper.patience = self.newPatience
                                print('\nearly_stopper.patience is changed to: %s\n' % self.early_stopper.patience)
                        "Plot, update config and save config in log folder"
                        self.log_results(prefix)
                        if self.agent == 'SAC':
                            self.entropyPi_sum = 0
                            self.alpha_sum = 0
                        print('  end RL evaluation')
                    self.elapsed_epochs += 1
                if MPI.COMM_WORLD.Get_rank() == 0:
                    if self.early_stopper.early_stop:
                        self.config['early_stop'] = True
                        "Save last models"
                        self.save_chekpoint(prefix='last')
                        "Save last config in log folder"
                        self.saveConfig()
                        break

            if not self.RL_training:
                if self.doneAlive:
                    self.RL_training = self.buffer.current_size >= self.config['bufferCapacity'] // 2
                else:
                    self.RL_training = self.buffer.current_size >= self.buffer.numEpBuffer // 2
                if self.RL_training:
                    self.enoughRollout = True
                    print('********************* Enough rollouts to start RL_training *********************')
        return

    def train(self, training=True):
        self.actor_network.train(training)
        self.critic_network.train(training)

    def to_device(self, device='cpu'):
        torchDevice = torch.device(device)
        self.actor_network.to(torchDevice)
        self.critic_network.to(torchDevice)
        self.critic_target_network.to(torchDevice)
