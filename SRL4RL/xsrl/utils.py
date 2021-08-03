
import torch
import numpy as np
import os
import cv2
import gc

from SRL4RL import SRL4RL_path
from SRL4RL.utils.utilsPlot import visualizeMazeExplor, plot_xHat, plotEmbedding
from SRL4RL.utils.utilsEnv import reset_stack, add_noise, update_video, tensor2image, NCWH2WHC, render_env
from SRL4RL.utils.nn_torch import pytorch2numpy, numpy2pytorch, save_model
from SRL4RL.utils.utils import loadPickle, createFolder
from SRL4RL.xsrl.arguments import is_with_discoveryPi
from SRL4RL.rl.utils.runner import StateRunner

np2torch = lambda x, device: numpy2pytorch(x, differentiable=False, device=device)


def omega_last_layer(x):
    return torch.sigmoid(x)


def sampleNormal(mu, sig):
    noise = torch.randn_like(mu)
    return mu + noise * sig, noise


def resetState(obs, alpha, beta, gamma, config):
    device = torch.device(config['device'])
    if len(obs.shape) > 3:
        numEnv = obs.shape[0]
    else:
        numEnv = 1
    state = np.random.normal(0, 0.02, [numEnv, config['state_dim']])
    # do not add noise at reset! obs = add_noise(obs)
    state = initState(numEnv, state, np2torch(obs, device), alpha, beta, gamma, config)
    return state

def init_action(size, config):
    return np.zeros((size, config['action_dim']))

def initState(size, states, x, alpha, beta, gamma, config):
    device = torch.device(config['device'])
    with torch.no_grad():
        actions = init_action(size, config)
        # Compute state
        o_alpha = alpha(x)
        o_beta = beta(torch.cat((np2torch(states, device), np2torch(actions, device)), dim=1))
        input_gamma = torch.cat((o_alpha, o_beta), dim=1)
        states = pytorch2numpy(gamma(input_gamma))
    return states


def update_target_network(target, source, device=None):
    if device:
        source.to('cpu')
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    if device:
        source.to(device)
    return target


def normalizePi(pi, logPi, mu):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    # action_max = envEval.action_space.high[0]
    # action_min = envEval.action_space.low[0]
    # action_scale = torch.tensor((action_max - action_min).item() / 2.)
    # action_bias = torch.tensor((action_max + action_min) / 2.)
    action_scale = 1
    action_bias = 0
    mu = torch.tanh(mu) * action_scale + action_bias
    pi = torch.tanh(pi)
    epsilon = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
    LogPi_jacobian = torch.log(action_scale * (1 - pi.pow(2)) + epsilon).sum(-1, keepdim=True)
    logPi -= LogPi_jacobian
    pi = pi * action_scale + action_bias
    return pi, logPi, mu, LogPi_jacobian


def gaussian_logprob(noise, log_sig):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_sig).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def policy_last_layer_op(s, pi_head, mu_tail, log_sig_tail, config):
    head_out = pi_head(s)
    mu = mu_tail(head_out)
    log_sig_min = - 10  # before: - config['action_dim'] * norm
    log_sig_max = 2  # before: 12 * norm

    log_sig = log_sig_tail(head_out)  # +3
    log_sig = torch.clamp(log_sig, min=log_sig_min, max=log_sig_max)

    sig = log_sig.exp()
    assert not torch.isnan(log_sig).any().item(), 'isnan in log_sig!!'
    log_sig_detach = log_sig
    # for repameterization trick (mu + sig * N(0,1))
    x_t, noise = sampleNormal(mu=mu, sig=sig)
    logPi = gaussian_logprob(noise, log_sig)
    pi, logPi, mu, LogPi_jacobian = normalizePi(x_t, logPi, mu)

    assert not torch.isnan(head_out).any().item(), 'isnan in head_out!!'
    assert not torch.isnan(mu).any().item(), 'isnan in mu!!'

    return pi, logPi, log_sig_detach, mu, LogPi_jacobian.detach()


def policy_last_layer(s, pi_head, mu_tail, log_sig_tail, config, s_dvt=None, pi_head_dvt=None, mu_tail_dvt=None,
                      log_sig_tail_dvt=None, save_pi_logs=False):
    if s_dvt is not None:
        pi_dvt, logPi_dvt, _, _, _ = policy_last_layer_op(s_dvt, pi_head_dvt, mu_tail_dvt, log_sig_tail_dvt,
                                                          config)
    pi, logPi, log_sig, mu, LogPi_jacobian = policy_last_layer_op(s, pi_head, mu_tail, log_sig_tail,
                                                                  config)

    if save_pi_logs and (s_dvt is None):
        return pi, logPi, log_sig.detach(), mu.detach(), LogPi_jacobian.detach()
    elif save_pi_logs and (s_dvt is not None):
        return pi, logPi, pi_dvt, logPi_dvt, log_sig.detach(), mu.detach(), LogPi_jacobian.detach()
    else:
        return pi


def XSRL_nextObsEval(alpha, beta, gamma, omega, config, save_dir, gradientStep=None,
                       saved_step=None, suffix='last', debug=False):
    eval = suffix == 'eval'
    if eval:
        path_eval = os.path.join(save_dir, 'eval2obs')
        createFolder(path_eval, "eval2obs already exist")
    actionRepeat = config['actionRepeat']

    datasetEval_path = 'testDatasets/testDataset_{}'.format(config['new_env_name'])
    if actionRepeat > 1:
        datasetEval_path += '_noRepeatAction'
    elif config['distractor']:
        datasetEval_path += '_withDistractor'
    datasetEval_path += '.pkl'

    datasetEval_path = os.path.join(SRL4RL_path, datasetEval_path)
    dataset = loadPickle(datasetEval_path)
    actions, observations, measures = dataset['actions'], dataset['observations'], dataset['measures']
    # if debug:
    #     last_index = actionRepeat * 200
    #     actions, observations, measures = actions[:-last_index], observations[:-last_index], measures[:-last_index]

    measures = measures[1:][actionRepeat:][::actionRepeat]
    "force the Garbage Collector to release unreferenced memory"
    del dataset
    gc.collect()

    device = torch.device(config['device'])
    Loss_obs = lambda x, y: torch.nn.MSELoss(reduction='sum')(x, y) / (x.shape[0] * config['n_stack'])
    loss_log = 0

    print('  XSRL_nextObsEval (predicting next obs with PIeval_dataset) ......')
    eval_steps = None
    if config['new_env_name'] == 'TurtlebotMazeEnv':
        xHat_nextObsEval_step = 84
        eval_steps = [87, 88, 101, 115, 117, 439, 440]
    elif config['new_env_name'] == 'HalfCheetahBulletEnv':
        xHat_nextObsEval_step = 119
    elif config['new_env_name'] == 'InvertedPendulumSwingupBulletEnv':
        xHat_nextObsEval_step = 45
    elif config['new_env_name'] == 'ReacherBulletEnv':
        xHat_nextObsEval_step = 42
        eval_steps = [14, 25, 396]

    video_path = os.path.join(save_dir, 'piEval_{}.mp4'.format(suffix))
    if config['new_env_name'] == 'TurtlebotMazeEnv':
        fps = 5
    elif actionRepeat > 1:
        fps = 20 // actionRepeat
    else:
        fps = 5
    video_out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(int(588 * 2), 588)) if \
        config['color'] else \
        cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize=(int(588 * 2), 588), isColor=0)

    "init state with obs without noise"
    if config['n_stack'] > 1:
        nc = 3
        observation = reset_stack(observations[0][None], config)
        next_observation = reset_stack(observations[0][None], config)
    else:
        observation = observations[0][None]
    with torch.no_grad():
        stateExpl = resetState(observation, alpha, beta, gamma, config)

    step_rep = 0
    elapsed_steps = 0
    len_traj = (len(observations) - 1) // actionRepeat - 1
    assert len_traj == len(measures), 'wrong division in len_traj'
    all_states = np.zeros([len_traj, config['state_dim']])
    "observations[1:] -> remove reset obs and first actionRepeat time steps"
    for step, (pi, next_obs) in enumerate(zip(actions, observations[1:])):
        "Make a step"

        if config['n_stack'] > 1:
            if (step_rep + 1) > (config['actionRepeat'] - config['n_stack']):
                next_observation[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = next_obs
        elif (step_rep + 1) == config['actionRepeat']:
            next_observation = next_obs[None]
        step_rep += 1
        if ((step + 1) % actionRepeat == 0) and (step + 1) > actionRepeat:
            # (step + 1) > actionRepeat: let one iteration to better bootstrap the state estimation
            step_rep = 0
            TensA = numpy2pytorch(pi, differentiable=False, device=device).unsqueeze(dim=0)
            "predict next states"
            with torch.no_grad():
                o_alpha = alpha(np2torch(observation, device))
                o_beta = beta(torch.cat((np2torch(stateExpl, device), TensA), dim=1))
                input_gamma = torch.cat((o_alpha, o_beta), dim=1)
                s_next = gamma(input_gamma)
                "Reconstruct observations of current elapsed_steps for all trajectories"
                xHat = omega_last_layer(omega(s_next))
                loss_log += pytorch2numpy(Loss_obs(xHat, np2torch(next_observation, device)))

            "update video"
            update_video(im=255 * NCWH2WHC(next_observation[:, -3:, :, :]), color=config['color'], video_size=588,
                         video=video_out, fpv=config['fpv'], concatIM=255 * tensor2image(xHat[:, -3:, :, :]))

            if type(eval_steps) is list:
                saveIm = elapsed_steps in [xHat_nextObsEval_step] + eval_steps
                name_ = 'xHat_nextObsEval{}'.format(elapsed_steps)
            else:
                saveIm = elapsed_steps == xHat_nextObsEval_step
                name_ = 'xHat_nextObsEval'
            if saveIm:
                "plot image to check the image prediction quality"
                if config['n_stack'] > 1:
                    "saving other frames"
                    for step_r in range(config['n_stack']):
                        name = 'xHat_nextObsEval{}_frame{}'.format(elapsed_steps, step_r)
                        plot_xHat(NCWH2WHC(observation[:, step_r * nc: (step_r + 1) * nc]),
                                  tensor2image(xHat[:, step_r * nc: (step_r + 1) * nc]),
                                  imgTarget=NCWH2WHC(next_observation[:, step_r * nc: (step_r + 1) * nc]),
                                  figure_path=save_dir,
                                  with_nextObs=True, name=name, gradientStep=gradientStep,
                                  suffix=suffix, eval=eval)
                else:
                    plot_xHat(NCWH2WHC(observation[:, -3:, :, :]),
                              tensor2image(xHat[:, -3:, :, :]),
                              imgTarget=NCWH2WHC(next_observation[:, -3:, :, :]),
                              figure_path=save_dir,
                              with_nextObs=True, name=name_,
                              gradientStep=gradientStep, suffix=suffix, eval=eval)
                if elapsed_steps == xHat_nextObsEval_step:
                    if saved_step is not None:
                        plot_xHat(NCWH2WHC(observation[:, -3:, :, :]),
                                  tensor2image(xHat[:, -3:, :, :]),
                                  imgTarget=NCWH2WHC(next_observation[:, -3:, :, :]),
                                  figure_path=os.path.join(save_dir, 'xHat_nextObsEval'),
                                  with_nextObs=True, name='xHat_nextObsEval',
                                  gradientStep=gradientStep, saved_step=saved_step)
            if eval:
                "plot image of all time steps"
                plot_xHat(NCWH2WHC(observation[:, -3:, :, :]), tensor2image(xHat[:, -3:, :, :]),
                          imgTarget=NCWH2WHC(next_observation[:, -3:, :, :]),
                          figure_path=path_eval, with_noise=config['with_noise'],
                          with_nextObs=True, saved_step=elapsed_steps)
            "save state"
            all_states[elapsed_steps] = stateExpl[0]
            elapsed_steps += 1
            "update states"
            stateExpl = pytorch2numpy(s_next)
            "update inputs without noise for test"
            # observation = add_noise(next_observation.copy(), noise_adder, config)
            observation = next_observation.copy()
        elif ((step + 1) % actionRepeat == 0) and (step + 1) == actionRepeat:
            step_rep = 0
            observation = next_observation.copy()

    "Release everything if job is finished"
    video_out.release()
    cv2.destroyAllWindows()

    loss_logNorm = loss_log / len_traj
    print(' ' * 100 + 'done : nextObsEval = {:.3f}'.format(loss_logNorm))

    plotEmbedding('UMAP', measures.copy(), all_states, figure_path=save_dir, gradientStep=gradientStep,
                  saved_step=saved_step, proj_dim=3, suffix=suffix, env_name=config['env_name'], eval=eval)
    plotEmbedding('PCA', measures, all_states, figure_path=save_dir, gradientStep=gradientStep,
                  saved_step=saved_step,
                  proj_dim=3, suffix=suffix, env_name=config['env_name'], eval=eval)
    "force the Garbage Collector to release unreferenced memory"
    del actions, observations, measures, video_out, all_states, stateExpl, s_next, observation, next_observation, xHat
    gc.collect()
    return loss_logNorm


def piExplore2obs(envExplor, noise_adder, alpha, beta, gamma, omega, pi_head, mu_tail, log_sig_tail, config, save_dir,
                  suffix='last', debug=False,  eval=False, saved_step=None):
    device = torch.device(config['device'])
    with_discoveryPi = is_with_discoveryPi(config)
    if saved_step is None:
        saved_step = ''
    else:
        saved_step = '_E{}'.format(saved_step)
    if config['env_name'] in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:
        camera_id_eval = 1
        imLabel = 'map'
    else:
        camera_id_eval = -1
        imLabel = 'env'
    if eval:
        path_eval = os.path.join(save_dir, 'piExplore2obs{}/'.format(saved_step))
        createFolder(path_eval, "piExplore2obs already exist")
        path_eval_im = os.path.join(save_dir, 'piExplore2im{}/'.format(saved_step))
        createFolder(path_eval_im, "piExplore2im already exist")

    obs = envExplor.reset()
    "init state with obs without noise"
    if config['n_stack'] > 1:
        nc = 3
        actionRepeat = config['actionRepeat']
        observation = reset_stack(obs, config)
        next_observation = reset_stack(obs, config)
    else:
        actionRepeat = 1
        observation = obs
    with torch.no_grad():
        stateExpl = resetState(observation, alpha, beta, gamma, config)

    eval_steps = 30 if debug else 500
    video_path = os.path.join(save_dir, 'piExplore_{}{}.mp4'.format(suffix,saved_step))
    fps = 5
    video_out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=(int(588 * 2), 588)) if \
        config['color'] else \
        cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize=(int(588 * 2), 588), isColor=0)

    print('  piExplore2obs (exploring and predicting next obs) ......')
    for step in range(eval_steps):
        "Make a step"
        has_bump = True
        num_bump = 0
        while has_bump:
            if eval:
                assert num_bump < 500, "num_bump > 500"
            num_bump += 1
            if with_discoveryPi:
                "update policy distribution and sample action"
                with torch.no_grad():
                    TensA = policy_last_layer(np2torch(stateExpl, 'cpu'), pi_head, mu_tail, log_sig_tail,
                                              config=config).to(device)
                pi = pytorch2numpy(TensA.squeeze(dim=0))
            else:
                pi = envExplor.action_space.sample()
                TensA = numpy2pytorch(pi, differentiable=False, device=device).unsqueeze(dim=0)

            if config['bumpDetection']:
                has_bump = envExplor.bump_detection(pi)
            else:
                has_bump = False

        "Make a step"
        for step_rep in range(actionRepeat):
            obs, _, done, _ = envExplor.step(pi)
            if config['n_stack'] > 1:
                if (step_rep + 1) > (config['actionRepeat'] - config['n_stack']):
                    next_observation[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = obs
            elif (step_rep + 1) == actionRepeat:
                assert step_rep < 2, 'actionRepeat is already performed in env'
                next_observation = obs
        with torch.no_grad():
            "predict next states"
            o_alpha = alpha(np2torch(observation, device))
            o_beta = beta(torch.cat((np2torch(stateExpl, device), TensA), dim=1))
            input_gamma = torch.cat((o_alpha, o_beta), dim=1)
            s_next = gamma(input_gamma)

            "Reconstruct observations of current step for all trajectories"
            xHat = omega_last_layer(omega(s_next))

        "update video"
        update_video(im=255 * NCWH2WHC(next_observation[:, -3:, :, :]), color=config['color'], video_size=588,
                     video=video_out, fpv=config['fpv'], concatIM=255 * tensor2image(xHat[:, -3:, :, :]))
        if eval:
            im_high_render = render_env(envExplor, 256, False, camera_id_eval, config['color'],
                                        downscaling=False) / 255.
            plot_xHat(NCWH2WHC(observation[:, -3:, :, :]), tensor2image(xHat[:, -3:, :, :]),
                      imgTarget=NCWH2WHC(next_observation[:, -3:, :, :]), im_high_render=im_high_render, imLabel=imLabel,
                      figure_path=path_eval, with_noise=config['with_noise'],
                      with_nextObs=True, saved_step=step)
            im_high_render = render_env(envExplor, 588, False, camera_id_eval, config['color'],
                                        downscaling=False)
            cv2.imwrite(path_eval_im + 'ob_{:05d}'.format(step) + '.png', im_high_render[:, :, ::-1].astype(np.uint8))

        "update inputs without noise for test"
        # observation = add_noise(next_observation.copy(), noise_adder, config)
        observation = next_observation.copy()
        stateExpl = pytorch2numpy(s_next)

    "Release everything if job is finished"
    video_out.release()
    cv2.destroyAllWindows()
    "force the Garbage Collector to release unreferenced memory"
    del video_out, stateExpl, s_next, observation, next_observation, xHat
    gc.collect()


def getPiExplore(envExplor, noise_adder, alpha, beta, gamma, pi_head, mu_tail, log_sig_tail, config, save_dir,
                 n_epoch=None,debug=False, eval=False, suffix = ''):
    assert config['env_name'] in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0'], 'getPiExplore only with Turtlebot'
    device = torch.device(config['device'])
    with_discoveryPi = is_with_discoveryPi(config)

    observation = envExplor.reset()
    with torch.no_grad():
        stateExpl = resetState(observation, alpha, beta, gamma, config)

    if debug:
        eval_steps = [50, 100]
    elif config['env_name'] == 'TurtlebotEnv-v0':
        eval_steps = [100, 200, 300]
    elif config['env_name'] == 'TurtlebotMazeEnv-v0':
        eval_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    robot_pos = np.zeros((eval_steps[-1] + 1, 2))
    eval_i = 0
    robot_pos[0] = envExplor.object.copy()

    if n_epoch:
        n_epoch_ = '-%06d' % n_epoch
    else:
        n_epoch_ = ''

    print('  getPiExplore (exploring) ......')
    for step in range(eval_steps[-1]):
        "Make a step"
        has_bump = True
        num_bump = 0
        while has_bump:
            if eval:
                assert num_bump < 500, "num_bump > 500"
            num_bump += 1
            if with_discoveryPi:
                "update policy distribution and sample action"
                with torch.no_grad():
                    TensA = policy_last_layer(np2torch(stateExpl, 'cpu'), pi_head, mu_tail, log_sig_tail,
                                              config=config).to(device)
                pi = pytorch2numpy(TensA.squeeze(dim=0))
            else:
                pi = envExplor.action_space.sample()
                TensA = numpy2pytorch(pi, differentiable=False, device=device).unsqueeze(dim=0)

            if config['bumpDetection']:
                has_bump = envExplor.bump_detection(pi)
            else:
                has_bump = False

        "Make a step"
        obs, _, done, _ = envExplor.step(pi)
        "store robot pos"
        robot_pos[step + 1] = envExplor.object.copy()
        if (step + 1) == eval_steps[eval_i]:
            visualizeMazeExplor(config['env_name'], robot_pos=robot_pos[:eval_steps[eval_i]].copy(), save_dir=save_dir,
                                name='explore{}{}{}'.format(eval_steps[eval_i] , n_epoch_, suffix))
            eval_i += 1

        next_observation = obs
        "predict next states"
        with torch.no_grad():
            o_alpha = alpha(np2torch(observation, device))
            o_beta = beta(torch.cat((np2torch(stateExpl, device), TensA), dim=1))
            input_gamma = torch.cat((o_alpha, o_beta), dim=1)
            s_next = gamma(input_gamma)

        "update inputs without noise for test"
        # observation = add_noise(next_observation.copy(), noise_adder, config)
        observation = next_observation
        stateExpl = pytorch2numpy(s_next)
    "force the Garbage Collector to release unreferenced memory"
    del robot_pos, s_next, stateExpl, observation, next_observation
    gc.collect()


class XSRLRunner(StateRunner):
    def __init__(self, config):
        super().__init__(config)
        self.alpha, self.beta, self.gamma = torch.load(
            os.path.join(config['srl_path'], 'state_model.pt'), map_location=torch.device('cpu'))
        self.alpha.eval(), self.beta.eval(), self.gamma.eval()

        self.initState()

    def resetState(self):
        self.state = self.initState().to('cpu')
        self.pi = np.zeros((self.action_dim))

    def update_state(self, x, demo=False):
        with torch.no_grad():
            "predict next state"
            input = add_noise(x, self.noise_adder, self.noiseParams)
            o_alpha = self.alpha(input.to(self.device)).to('cpu')
            "FNNs only faster with cpu"
            o_beta = self.beta(torch.cat((self.state, np2torch(self.pi, 'cpu').unsqueeze(0)), dim=1))
            input_gamma = torch.cat((o_alpha, o_beta), dim=1)
            new_state = self.gamma(input_gamma)
        if demo:
            self.last_input = pytorch2numpy(input)[0][-3:, :, :].transpose(1, 2, 0)
        self.state = new_state
        return new_state


    def save_state_model(self, save_path):
        print('Saving models ......')
        save_model([self.alpha, self.beta, self.gamma], save_path + 'state_model')

    def train(self, training=True):
        self.alpha.train(training)
        self.beta.train(training)
        self.gamma.train(training)

    def to_device(self, device='cpu'):
        torchDevice = torch.device(device)
        self.alpha.to(torchDevice)
        self.beta.to('cpu')
        self.gamma.to('cpu')



