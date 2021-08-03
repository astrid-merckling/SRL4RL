import os
import torch
import numpy as np
import time
from datetime import timedelta

from bullet_envs.utils import env_with_goals

from SRL4RL.utils.utils import float2string, saveConfig, saveJson, EarlyStopping, update_text, encoder_methods, state_baselines
from SRL4RL.utils.nn_torch import numpy2pytorch, save_model
from SRL4RL.rl.utils.replay_buffer import sample_transitions
from SRL4RL.utils.utilsPlot import plotter


def loadPi(path, model_type = 'model_last', withQ=False):
    "Load RL model"
    o_mean, o_std, g_mean, g_std, actor_network, critic_network = torch.load(os.path.join(path, model_type+'.pt'),
                                                             map_location=lambda storage, loc: storage)
    actor_network.eval()
    print('Load Pi')
    if withQ:
        critic_network.eval()
        return o_mean, o_std, g_mean, g_std, actor_network, critic_network
    else:
        return o_mean, o_std, g_mean, g_std, actor_network

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, config):
    if (np.abs(o) > config['clip_obs']).any():
        print('\nstate is HUGE\n')
        o = np.clip(o, -config['clip_obs'], config['clip_obs'])
    o_norm = np.clip((o - o_mean) / o_std, -config['clip_range'], config['clip_range'])
    inputs = o_norm
    if config['with_goal']:
        g_norm = np.clip((g - g_mean) / g_std, -config['clip_range'], config['clip_range'])
        inputs = np.concatenate([o_norm, g_norm])
    inputs = numpy2pytorch(inputs, differentiable=False).unsqueeze(0)
    return inputs

def process_inputs_gt(o, g, o_mean, o_std, g_mean, g_std, clip_obs = 200, clip_range = 10, with_goal= False):
    if (np.abs(o) > clip_obs).any():
        print('\nstate is HUGE\n')
        o = np.clip(o, -clip_obs, clip_obs)
    o_norm = np.clip((o - o_mean) / o_std, -clip_range, clip_range)
    inputs = o_norm
    if with_goal:
        g_norm = np.clip((g - g_mean) / g_std, -clip_range, clip_range)
        inputs = np.concatenate([o_norm, g_norm])
    inputs = numpy2pytorch(inputs, differentiable=False).unsqueeze(0)
    return inputs


class Trainer():
    def __init__(self, ):
        pass

    def _preproc_og(self, o):
        if torch.is_tensor(o):
            o = torch.clamp(o, -self.clip_obs, self.clip_obs)
        else:
            o = np.clip(o, -self.clip_obs, self.clip_obs)
        return o

    # pre_process the inputs
    def _preproc_inputs(self, obs, g=None, device=torch.device('cpu')):
        if (np.abs(obs) > self.clip_obs).any():
            print('\nstate is HUGE\n')
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        obs_norm = self.o_norm.normalize(obs)
        if self.with_goal:
            g_norm = self.g_norm.normalize(g)
            obs_norm = np.concatenate([obs_norm, g_norm])
        obs_norm = numpy2pytorch(obs_norm, differentiable=False, device=device).unsqueeze(0)
        return obs_norm

    def normalize_goal(self, g, device=torch.device('cpu')):
        g_norm = self.g_norm.normalize(g)
        g_tensor = numpy2pytorch(g_norm, differentiable=False, device=device)
        return g_tensor.unsqueeze(0)

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_g, mb_r, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        # create the new buffer to store them
        if self.with_goal:
            buffer_temp = {'obs': mb_obs,
                           'g': mb_g,
                           'r': mb_r,
                           'actions': mb_actions,
                           'obs_next': mb_obs_next,
                           }
        else:
            buffer_temp = {'obs': mb_obs,
                           'r': mb_r,
                           'actions': mb_actions,
                           'obs_next': mb_obs_next,
                           }
        transitions = sample_transitions(buffer_temp, batch_size_in_transitions=None)
        if self.with_goal:
            self.g_norm.update(transitions['g'])
            self.g_norm.recompute_stats()
        # pre process the obs and g
        transitions['obs'] = self._preproc_og(transitions['obs'])
        self.o_norm.update(transitions['obs'])
        self.o_norm.recompute_stats()

    def prepare_data(self, transitions, device=torch.device('cpu')):
        "pre-process the observation and goal"
        g_tensor, not_dones = None, 1
        transitions['obs'] = self._preproc_og(transitions['obs'])
        transitions['obs_next'] = self._preproc_og(transitions['obs_next'])
        obs_norm = self.o_norm.normalize(transitions['obs'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        if self.with_goal:
            g_norm = self.g_norm.normalize(transitions['g'])
            if torch.is_tensor(obs_norm):
                g_norm = numpy2pytorch(g_norm, differentiable=False, device=device)
                inputs_norm_tensor = torch.cat([obs_norm.to(device), g_norm], 1)
                inputs_next_norm_tensor = torch.cat([obs_next_norm.to(device), g_norm], 1)
            else:
                obs_norm = np.concatenate([obs_norm, g_norm], axis=1)
                obs_next_norm = np.concatenate([obs_next_norm, g_norm], axis=1)

        if not torch.is_tensor(obs_norm):
            # start to do the update, transfer them into the tensor
            inputs_norm_tensor = numpy2pytorch(obs_norm, differentiable=False, device=device)
            inputs_next_norm_tensor = numpy2pytorch(obs_next_norm, differentiable=False, device=device)
        if self.doneAlive:
            not_dones = numpy2pytorch(transitions['not_dones'], differentiable=False, device=device)

        actions_tensor = numpy2pytorch(transitions['actions'], differentiable=False, device=device)
        r_tensor = numpy2pytorch(transitions['r'], differentiable=False, device=device).reshape(-1, 1)

        return inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor, g_tensor, not_dones


class Saver(Trainer):
    def __init__(self, config, env):
        super().__init__()
        "Initialize early_stopper"
        self.min_reward = np.inf
        "one epoch corresponds to gradient_steps"
        "patience corresponds to min_gradientSteps = [patience*gradient_steps*eval_interval] before early-stopping"
        self.min_Nepochs = np.inf
        if self.env_name in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0', 'ReacherBulletEnv-v0']:
            self.min_reward = 1.
            self.newPatience = 400
            if config['method'] == 'random_nn':
                self.newPatience = config['patience'] = config['patience'] * 4
        else:
            self.newPatience = 50

        if config['method'] in ['pure_noise', 'openLoop']:
            self.newPatience = config['patience'] = config['patience'] * 4
        print('min_Nepochs: {}, min_reward: {}'.format(self.min_Nepochs, self.min_reward))
        self.config['early_stop'] = False

        if config['dir']:
            config['random_buffer'] = False
            prefix = 'last_' if 'last_eval' in config else 'best_'
            self.early_stopper = EarlyStopping(patience=config['patience'], name=self.hashCode,
                                               min_Nepochs=config['patience'] * self.eval_interval + config['{}elapsed_epochs'.format(prefix)])
            self.elapsed_epochs = config['{}elapsed_epochs'.format(prefix)]
            self.elapsed_gradients = config['{}elapsed_gradients'.format(prefix)]
            self.elapsed_soft_update = config['{}elapsed_soft_update'.format(prefix)]
            self.elapsed_rollouts = config['{}elapsed_rollouts'.format(prefix)]
            self.elapsed_steps = config['{}elapsed_steps'.format(prefix)]
            self.loaded_time_s = config['{}elapsed_time_s'.format(prefix)]
        else:
            self.early_stopper = EarlyStopping(patience=config['patience'], name=self.hashCode,
                                               min_Nepochs=config['patience'] * self.eval_interval)
            self.elapsed_epochs = self.elapsed_gradients = self.elapsed_soft_update = self.elapsed_rollouts = self.elapsed_steps = 0
            self.loaded_time_s = 0
        self.backprop_per_eval = config['gradient_steps'] * config['eval_interval']
        self.start_time = time.time()

        if self.env_name in env_with_goals:
            self.ylabel = 'success rate'
        else:
            self.ylabel = 'score'

        if config['with_images']:
            if self.method in encoder_methods:
                self.runner.save_state_model(self.save_dir)
            else:
                save_model(env.encoder, self.save_dir + self.method)

    def save_chekpoint(self, prefix=None):
        print('Saving models ......')
        prefix = '' if prefix is None else '_' + prefix
        if prefix in ['_last', '_best']:
            save_dir = self.save_dir + 'model' + prefix + '.pt'
        torch.save(
            [self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network,
             self.critic_network], save_dir)

    def saveConfig(self):
        saveConfig(self.config, save_dir=self.save_dir)
        saveJson(self.config, self.save_dir)

    def update_success(self, success_rate):
        update_text([success_rate], self.save_dir + "success_rates.txt")
        self.eval = success_rate
        "for early-stopper"
        with open(self.save_dir + "success_rates.txt") as f:
            loss_log = np.array(f.read().split("\n"), dtype=np.float32)
            "remove outliers"
            self.eval_patience_mean = np.mean((loss_log[-self.early_stopper.patience:]))
            loss_log[loss_log<np.percentile(loss_log,25)] = np.percentile(loss_log,25)
            self.eval_patience = np.mean((loss_log[-self.early_stopper.patience:]))
        if self.agent == 'SAC' and self.elapsed_epochs > 0:
            update_text([self.actor_update_interval * self.entropyPi_sum / (self.gradient_steps * self.eval_interval)],
                        self.save_dir + "entropyPi.txt")
            update_text([self.actor_update_interval * self.alpha_sum / (self.gradient_steps * self.eval_interval)],
                        self.save_dir + "alpha.txt")

    def update_config(self, prefix=None):
        self.config[prefix + 'elapsed_time_s'] = time.time() - self.start_time + self.loaded_time_s
        self.config[prefix + 'elapsed_time'] = str(timedelta(seconds=(int(self.config[prefix + 'elapsed_time_s']))))
        self.config[prefix + 'elapsed_epochs'] = self.elapsed_epochs
        self.config[prefix + 'elapsed_soft_update'] = self.elapsed_soft_update
        self.config[prefix + 'elapsed_gradients'] = self.elapsed_gradients
        self.config[prefix + 'elapsed_rollouts'] = self.elapsed_rollouts
        self.config[prefix + 'elapsed_steps'] = self.elapsed_steps
        self.config['eval-patience'] = float2string(self.eval_patience)
        self.config['eval-patience_mean'] = float2string(self.eval_patience_mean)
        self.config[prefix + 'eval'] = float2string(self.eval)

    def log_results(self, prefix):
        "Plot, update config and save config in log folder"
        prefix = '' if prefix is None else prefix + '_'

        plotter(self.eval, self.save_dir, name='success_rates',
                title="RL training \n{}".format(self.config['hashCode']),
                ylabel='Average {} over {} episodes'.format(self.ylabel, self.n_eval_rollouts),
                backprop_per_eval=self.backprop_per_eval, text_file=self.save_dir + "success_rates.txt")
        if self.elapsed_epochs > 0:
            plotter(0, self.save_dir, name='entropy',
                    title="RL training \n{}".format(self.config['hashCode']),
                    ylabel='Average entropy',
                    backprop_per_eval=self.backprop_per_eval, text_file=self.save_dir + "entropyPi.txt")
            plotter(0, self.save_dir, name='alpha',
                    title="RL training \n{}".format(self.config['hashCode']),
                    ylabel='Average alpha',
                    backprop_per_eval=self.backprop_per_eval, text_file=self.save_dir + "alpha.txt")

        self.update_config(prefix)
        self.saveConfig()

        print("{} \n[{}]  *{}+{}*  \n[{}] Epoch: {} | Gradient steps: {} | {}: {:.3f} | {}: {:.3f} | {}: {:.3f}".format(
            self.config['env_params_name'], self.config['hashCode'],  self.config['RL_name'], self.config['method'],self.config[prefix + 'elapsed_time'],
            self.elapsed_epochs, self.elapsed_gradients, self.ylabel, self.eval,
            'eval-patience',self.eval_patience,
            'eval-patience_mean',self.eval_patience_mean,))
        if self.config['method'] in encoder_methods:
            E_hashCode = '[{}]'.format(self.config['E_hashCode']) if 'E_hashCode' in self.config else ''
            print("   SRL {} {} [{}]".format(E_hashCode, self.config['method'], self.config['SRL_name']))
