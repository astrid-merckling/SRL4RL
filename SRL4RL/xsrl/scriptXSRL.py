"""
05/03/2020
Astrid Merckling
XSRL
"""
from pprint import pprint
import matplotlib
import numpy as np
import gym
import torch
import os
import json, hashlib
from datetime import datetime, timedelta
from time import time
from collections import OrderedDict
import gc

from bullet_envs.utils import PY_MUJOCO, AddNoise

from SRL4RL.utils.nn_torch import CNN, CNN_Transpose, MLP_mdn, MLP_Module, pytorch2numpy, numpy2pytorch, set_seeds, save_model
from SRL4RL.utils.utils import createFolder, saveConfig, EarlyStopping, loadConfig, get_hidden, update_text, float2string, saveJson
from SRL4RL.utils.env_wrappers import BulletWrapper
from SRL4RL.utils.utilsEnv import add_noise, render_env, giveEnv_name, reset_stack
from SRL4RL import SRL4RL_path

from SRL4RL.utils.utilsPlot import plot_xHat, plotter

from SRL4RL.xsrl.utils import policy_last_layer, omega_last_layer, getPiExplore, piExplore2obs, resetState, \
    XSRL_nextObsEval, update_target_network
from SRL4RL.xsrl.arguments import is_with_discoveryPi, get_args, update_args_XSRL, giveXSRL_name, assert_args_XSRL

"register bullet_envs in gym"
import bullet_envs.__init__

# get the params
args = get_args()
args = update_args_XSRL(args)

"Pretraining setup"
all_dir = None
if args.dir:
    if args.dir[-1] != '/': args.dir += '/'
    loaded_config = loadConfig(args.dir)
    dir_hashCode = loaded_config['hashCode']
    remove_keys = ['keep_seed', 'dir', 'debug', 'reset_policy', 'patience', 'n_epochs']
    if args.keep_seed:
        assert loaded_config['hashCode'] == dir_hashCode
        if 'all_dir' in loaded_config:
            all_dir = loaded_config['all_dir']
    else:
        remove_keys += ['logs_dir', 'seed', 'flickering', 'wallDistractor', 'noise_type', 'lr', 'lr_explor',
                      'lr_alpha', 'backprop_per_eval', 'resetPi', 'dvt_patience', 'weightEntropy', 'weightLPB',
                      'weightInverse', 'autoEntropyTuning', 'pi_batchSize']
        if 'all_dir' not in loaded_config:
            all_dir = dir_hashCode
        else:
            all_dir = loaded_config['all_dir'] + '-' + dir_hashCode

    dir_inverse = loaded_config['inverse']
    if args.keep_seed:
        select_new_args = {k: args.__dict__[k] for k in remove_keys}
        loaded_config.update(select_new_args)
        del select_new_args
    else:
        keep_keys = list(args.__dict__.keys())
        for k in remove_keys:
            keep_keys.remove(k)
        keep_keys += ['n_stack']
        select_old_args = {k: loaded_config[k] for k in keep_keys}
        args.__dict__.update(select_old_args)
        del keep_keys, select_old_args

    "force the Garbage Collector to release unreferenced memory"
    gc.collect()

maxStep_eval = 500
if args.debug:
    hidden_beta = [2, 2, 2]
    args.nb_hidden_gamma = '2-2-2'
    args.nb_hidden_pi = '2-2-2'
    "We use num_envs parallel envs for data collection while training"
    args.num_envs = 4
    args.pi_batchSize = 8
    args.pi_updateInt = 16
    args.backprop_per_eval = 32
    args.maxStep = 30
else:
    hidden_beta = [128, 256, 32]
    matplotlib.use('Agg')

args = assert_args_XSRL(args)


"Training settings"
cpu = torch.device('cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
args.n_epochs = int(args.n_epochs)
with_discoveryPi = is_with_discoveryPi(args.__dict__)

np2torch = lambda x: numpy2pytorch(x, differentiable=False, device=cpu)
np2torchDev = lambda x: numpy2pytorch(x, differentiable=False, device=device)

if args.evalExplor:
    "to eval on a max of 500 steps"
    args.maxStep -= args.num_envs

if args.maxStep == np.inf:
    maxSteps = [args.maxStep] * args.num_envs
else:
    "to remove correlations between each robot trajectory"
    maxSteps = np.linspace(args.maxStep, args.maxStep + args.num_envs - 1, args.num_envs).astype(int)
    print('maxSteps: ', maxSteps)

"Environment settings"
evalSteps = 5  # to evaluate reconstructed images

maxSteps_visuExplor_env = 1000  # good because bigger than len(testDataset)=400
if args.seed == 123456:
    args.seed = datetime.now().microsecond
nc = 3 if args.color else 1
image_size = [nc, args.image_size, args.image_size]

seed_testDataset = 115
"noise added in add_noise(), not in real env"


if args.env_name in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:

    camera_id_eval = 1
    imLabel = 'map'
    actionRepeat = args.actionRepeat
    assert args.actionRepeat == 1
    visuExplor_env = gym.make(args.env_name, renders=False,
                              maxSteps=maxSteps_visuExplor_env, actionRepeat=args.actionRepeat,
                              image_size=image_size[-1], color=True, seed=seed_testDataset,
                              with_target=False, noise_type='none', fpv=args.fpv, display_target=False,
                              randomExplor=False,
                              wallDistractor=args.wallDistractor, distractor=args.distractor)

    envs = [gym.make(args.env_name, renders=False, maxSteps=maxSteps[i_], actionRepeat=args.actionRepeat,
                     image_size=image_size[-1], color=True, seed=args.seed + i_ + 1,
                     with_target=False, noise_type='none', fpv=args.fpv, display_target=False,
                     randomExplor=False,
                     wallDistractor=args.wallDistractor, distractor=args.distractor) for i_ in range(args.num_envs)]

    envEval = gym.make(args.env_name, renders=False, maxSteps=maxStep_eval, actionRepeat=args.actionRepeat,
                       image_size=image_size[-1], color=True, seed=args.seed,
                       with_target=False, noise_type='none', fpv=args.fpv, display_target=False,
                       randomExplor=False,
                       wallDistractor=args.wallDistractor, distractor=args.distractor)

elif args.env_name in PY_MUJOCO:

    camera_id_eval = -1
    imLabel = 'env'
    evalSteps = evalSteps
    "actionRepeat is done only one time"
    if args.n_stack > 1:
        actionRepeat = args.actionRepeat
        actionRepeat_env = 1
    else:
        actionRepeat = 1
        actionRepeat_env = args.actionRepeat
    visuExplor_env = gym.make('PybulletEnv-v0', env_name=args.env_name, renders=False,
                              maxSteps=maxSteps_visuExplor_env * args.actionRepeat, actionRepeat=actionRepeat_env,
                              image_size=image_size[-1], color=True, seed=seed_testDataset,
                              noise_type='none', fpv=args.fpv, doneAlive=False, randomExplor=False,
                              distractor=args.distractor)
    envs = [gym.make('PybulletEnv-v0', env_name=args.env_name, renders=False, maxSteps=maxSteps[i_] * args.actionRepeat,
                     actionRepeat=actionRepeat_env,
                     image_size=image_size[-1], color=True, seed=args.seed + i_ + 1,
                     noise_type='none', fpv=args.fpv, doneAlive=False, randomExplor=False,
                     distractor=args.distractor) for i_ in
            range(args.num_envs)]
    envEval = gym.make('PybulletEnv-v0', env_name=args.env_name, renders=False,
                       maxSteps=maxStep_eval * args.actionRepeat, actionRepeat=actionRepeat_env,
                       image_size=image_size[-1], color=True, seed=args.seed,
                       noise_type='none', fpv=args.fpv, doneAlive=False, randomExplor=False,
                       distractor=args.distractor)

visuExplor_env.seed(seed_testDataset)
[env_.seed(args.seed + i_ + 1) for i_, env_ in enumerate(envs)]
envEval.seed(args.seed)

"wrap envs"
visuExplor_env = BulletWrapper(visuExplor_env, args.__dict__)
envs = [BulletWrapper(env_, args.__dict__) for env_ in envs]
envEval = BulletWrapper(envEval, args.__dict__)

action_dim = envEval.action_space.shape[0]

"Create config"
if args.keep_seed:
    config = loaded_config
else:
    config = OrderedDict(sorted(args.__dict__.items()))
    config['hashCode'] = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['date'] = datetime.now().strftime("%y-%m-%d_%Hh%M_%S")
    config['action_dim'] = action_dim
    config['device'] = device
    config['with_noise'] = args.noise_type != 'none'

hashCode = config['hashCode']
env_params_name = config['new_env_name'] + ' ' + giveEnv_name(config)
with_noise = config['with_noise']
if with_noise:
    noise_adder = AddNoise(config)
else:
    noise_adder = None

if all_dir: config['all_dir'] = all_dir
if args.keep_seed:
    save_path = args.dir
else:
    save_path = os.path.join(SRL4RL_path, args.logs_dir)
    "Create folder, save config"
    if save_path[-1] != '/': save_path += '/'
    save_path += '%s' % hashCode + '/'
    createFolder(save_path, "save_path folder already exist")
    createFolder(save_path + 'xHat', "xHat folder already exist")
    createFolder(save_path + 'xHat_nextObsEval', "xHat_nextObsEval folder already exist")
    createFolder(save_path + 'UMAPproj', "UMAPproj folder already exist")
    createFolder(save_path + 'PCAproj', "PCAproj folder already exist")

config = OrderedDict(sorted(config.items()))
pprint(config)


"Initialize models"
if args.beta_dim == 0:
    args.beta_dim = args.state_dim + action_dim
reset_policy, reset_inverse = True, True

if args.dir:
    print('Load: alpha, beta, gamma, omega')
    alpha, beta, gamma = torch.load(args.dir + 'state_model.pt', map_location=torch.device(device))
    omega = torch.load(args.dir + 'state_model_tail.pt', map_location=torch.device(device))
    if with_discoveryPi and (not args.reset_policy):
        print('Load: pi_head, mu_tail, log_sig_tail')
        reset_policy = False
        pi_head, mu_tail, log_sig_tail = torch.load(args.dir + 'piExplore.pt', map_location=torch.device(cpu))
    if dir_inverse and not args.reset_policy:
        print('Load: iota')
        reset_inverse = False
        iota = torch.load(args.dir + 'iota.pt', map_location=torch.device(cpu))
else:
    print('Create: alpha, beta, gamma, omega')
    "Alpha function: create features for an observation"
    alpha = CNN(args.alpha_dim, nc * config['n_stack'], activation=args.activation,
                debug=args.debug)
    "Beta function: predict action and state features"
    beta = MLP_Module(args.state_dim + action_dim, hidden_beta + [args.beta_dim], activation=args.activation)
    "Gamma function: predict next state"
    gamma = MLP_Module(args.beta_dim + args.alpha_dim, get_hidden(args.nb_hidden_gamma) + [args.state_dim],
                       activation=args.activation)
    "Omega function: predict next observation"
    omega = CNN_Transpose(args.state_dim, nc * config['n_stack'], probabilistic=False, cutoff=args.cutoff,activation=args.activation, debug=args.debug)

alpha.to(device).train(), beta.to(device).train(), gamma.to(device).train(), omega.to(device).train()


if with_discoveryPi and reset_policy:
    print('Create: pi_head, mu_tail, log_sig_tail')
    pi_head, mu_tail, log_sig_tail, _ = MLP_mdn(args.state_dim, get_hidden(args.nb_hidden_pi) + [action_dim],
                                                cutoff=args.cutoff,
                                                activation=args.activation)
if args.inverse and reset_inverse:
    print('Create: iota')
    iota = MLP_Module(args.state_dim * 2, get_hidden(args.nb_hidden_pi) + [action_dim],
                      activation=args.activation)

if args.resetPi:
    print('Create: pi_head_dvt, mu_tail_dvt, log_sig_tail_dvt')
    pi_head_dvt, mu_tail_dvt, log_sig_tail_dvt, _ = MLP_mdn(args.state_dim,
                                                            get_hidden(args.nb_hidden_pi) + [action_dim],
                                                            cutoff=args.cutoff,
                                                            activation=args.activation)
    pi_head_dvt.to(cpu).train(), mu_tail_dvt.to(cpu).train(), log_sig_tail_dvt.to(cpu).train()
    policyOptimizer_dvt = torch.optim.Adam(list(pi_head_dvt.parameters()) + list(mu_tail_dvt.parameters()) + list(
        log_sig_tail_dvt.parameters()), lr=args.lr_explor)
else:
    pi_head_dvt, mu_tail_dvt, log_sig_tail_dvt = None, None, None

if args.LPB:
    "prepare the old networks for LPB"
    alpha_old = CNN(args.alpha_dim, image_size[0] * config['n_stack'], activation=args.activation, requires_grad=False, debug=args.debug)
    beta_old = MLP_Module(args.state_dim + action_dim, hidden_beta + [args.beta_dim], activation=args.activation,
                          requires_grad=False)
    gamma_old = MLP_Module(args.beta_dim + args.alpha_dim, get_hidden(args.nb_hidden_gamma) + [args.state_dim],
                           activation=args.activation, requires_grad=False)
    alpha_old.to(cpu).train(), beta_old.to(cpu).train(), gamma_old.to(cpu).train()
    "Update old network with current network's weights"
    alpha_old = update_target_network(alpha_old, alpha, device=device)
    if device == 'cuda':
        beta.to('cpu'), gamma.to('cpu')
    beta_old = update_target_network(beta_old, beta)
    gamma_old = update_target_network(gamma_old, gamma)
    if device == 'cuda':
        beta.to(device), gamma.to(device)

parameters = list(alpha.parameters()) + list(beta.parameters()) + list(gamma.parameters()) + list(
    omega.parameters())

"create state estimator optimizer"
amsgrad = args.optimizer == 'amsgrad'
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
else:
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, amsgrad=amsgrad)
"force the Garbage Collector to release unreferenced memory"
del parameters
gc.collect()

if with_discoveryPi:
    "create discovery policy optimizer"
    pi_head.to(cpu).train(), mu_tail.to(cpu).train(), log_sig_tail.to(cpu).train()
    policyParameters = list(pi_head.parameters()) + list(mu_tail.parameters()) + list(log_sig_tail.parameters())

    if args.optimizer == 'adam':
        policyOptimizer = torch.optim.Adam(policyParameters, lr=args.lr_explor)
    else:
        policyOptimizer = torch.optim.AdamW(policyParameters, lr=args.lr_explor, amsgrad=amsgrad)
    "force the Garbage Collector to release unreferenced memory"
    del policyParameters
    gc.collect()

    if args.inverse:
        iota.to(cpu).train()
        if args.lr < 5e-5 and not args.reset_policy:
            lr_inverse = 5e-5
        else:
            lr_inverse = args.lr_explor
        print('lr_inverse is {}'.format(lr_inverse))
        if args.optimizer == 'adam':
            iotaOptimizer = torch.optim.Adam(list(iota.parameters()), lr=lr_inverse)
        else:
            iotaOptimizer = torch.optim.AdamW(list(iota.parameters()), lr=lr_inverse, amsgrad=amsgrad)
else:
    pi_head, mu_tail, log_sig_tail = None, None, None

early_stopper = EarlyStopping(patience=args.patience, name=args.method + '-{}'.format(hashCode), min_delta=0.0001,
                              min_Nepochs=2 * args.patience)
config['early_stop'] = False

"IMPORTANT TO USE FOR CUDA MEMORY"
set_seeds(args.seed)

"Define losses"
Loss = lambda x, y, den, reduction='none': torch.nn.MSELoss(reduction=reduction)(x, y) / den
Loss_obs = lambda x, y, den, reduction='none': torch.nn.MSELoss(reduction=reduction)(x, y) / (den * config['n_stack'])
meanFctNorm = lambda x: (x ** 2).sum(1).sqrt().sum() / x.shape[0]
sumFct = lambda x: x.sum() / x.shape[0]

if args.autoEntropyTuning:
    target_entropy = np.float32(-action_dim)
    log_temperatureTensor = numpy2pytorch(np.array(np.log(args.init_temperature)), differentiable=True,
                                          device=device)
    "to force alpha is positive"
    temperatureTensor = log_temperatureTensor.exp()
    log_temperature_optim = torch.optim.Adam([log_temperatureTensor], lr=args.lr_alpha)


SRL_name = giveXSRL_name(config)
print('SRL_name: %s' % SRL_name)
if all_dir:
    print('all_dir: {}'.format(config['all_dir']))

k_pi = k_srl = loaded_time_s = 0

start_time = time()
lossNextObs_log = np.zeros((args.backprop_per_eval), np.float32)
if args.keep_seed:
    elapsed_steps = loaded_config['elapsed_steps']
    elapsed_gradients = loaded_config['elapsed_gradients']
    loaded_time_s = loaded_config['elapsed_time_s']
    elapsed_epochs = loaded_config['elapsed_epochs']
else:
    elapsed_gradients = elapsed_steps = elapsed_epochs = 0

pi_elapsed_gradients = 0
if with_discoveryPi:
    if args.keep_seed:
        pi_elapsed_gradients = loaded_config['pi_elapsed_gradients']
    DetLogCov_sum = CovNorm_sum = muNorm_sum = LogPi_jacobian_sum = 0
    DetLogCov_log, CovNorm_log, muNorm_log, LogPi_jacobian_log = [np.zeros((args.pi_backprop_per_eval), np.float32) for
                                                                  _ in
                                                                  range(4)]
    lossInverse_log, LPB_log, entropy_log, temperature_log = [np.zeros((args.pi_backprop_per_eval), np.float32) for _ in
                                                              range(4)]
    lossInverseSequential = r_entropySequential = r_inverseSequential = r_LPBsequential = 0

if args.resetPi:
    keep1stPolicy = False
    dvt_delta = 0.05
    f_stopper = lambda x: x * (1 - np.sign(x) * dvt_delta)
    dvt_counter = dvt_before2reset = dvtEquiv_before2reset = keep1stPolicy_counter = 0
    lossNextObs_dvt1_sum = lossNextObs_dvt2_sum = 0
    r_entropySequential_dvt = r_inverseSequential_dvt = r_LPBsequential_dvt = 0
    if args.weightInverse > 0 or args.weightLPB > 0:
        lossInverse_log_dvt, LPB_log_dvt = [np.zeros((args.pi_backprop_per_eval), np.float32) for _ in range(2)]
    elif args.weightEntropy > 0:
        entropy_log_dvt = np.zeros((args.pi_backprop_per_eval), np.float32)

if args.evalExplor:
    robot_pos = np.zeros((args.backprop_per_eval, args.num_envs, 2), np.float32)
    save_path_robotPos = save_path + 'robotPos/'
    createFolder(save_path_robotPos, "save_path_robotPos folder already exist")
    endDetection_dict = {}
    saveJson(endDetection_dict, save_path, 'endDetection.json')

if args.env_name in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:
    visuExplor_path = save_path + 'visuExplor/'
    createFolder(visuExplor_path, "visuExplor_path already exist")
else:
    visuExplor_path = ''

"reset reset info"
dones = [False] * args.num_envs
envs_to_reset = np.array(dones, dtype=np.bool)
reseted = np.arange(args.num_envs)[envs_to_reset]

if args.dir:
    del loaded_config
    gc.collect()

"init state with obs without noise"
if config['n_stack'] > 1:
    obs = np.vstack([env_.reset() for i_, env_ in enumerate(envs)])
    observations = reset_stack(obs, config)
    next_observations = reset_stack(obs, config)

    obs = envEval.reset()
    observation = reset_stack(obs, config)
    next_observation = reset_stack(obs, config)
else:
    observations = np.vstack([env_.reset() for i_, env_ in enumerate(envs)])
    observation = envEval.reset()


"Initialize states and observations"
states = resetState(observations, alpha, beta, gamma, config)
evalState = resetState(observation, alpha, beta, gamma, config)


torch.cuda.empty_cache()
"Start XSRL training"
stop_training = False
while not early_stopper.early_stop and elapsed_epochs < args.n_epochs and (not stop_training):
    "Do one step with every robots"
    elapsed_steps += 1

    "Make B robot steps"

    "with bump detection"
    has_bump_dvt = np.zeros((args.nEnv_perPi), dtype=np.bool)
    has_bump = np.ones((args.nEnv_perPi), dtype=np.bool)
    if with_discoveryPi:
        if args.resetPi:
            has_bump_dvt = np.ones((args.nEnv_perPi), dtype=np.bool)
            TensActions_dvt = torch.zeros((args.nEnv_perPi, action_dim), dtype=torch.float32).to('cpu')
            Tens_logPi_dvt = torch.zeros((args.nEnv_perPi, 1), dtype=torch.float32).to('cpu')
        TensActions = torch.zeros((args.nEnv_perPi, action_dim), dtype=torch.float32).to('cpu')
        Tens_logPi = torch.zeros((args.nEnv_perPi, 1), dtype=torch.float32).to('cpu')
        log_sig = torch.zeros((args.nEnv_perPi, action_dim), dtype=torch.float32).to('cpu')
        mu = torch.zeros((args.nEnv_perPi, action_dim), dtype=torch.float32).to('cpu')
        LogPi_jacobian = torch.zeros((args.nEnv_perPi, 1), dtype=torch.float32).to('cpu')
    else:
        ActEnv = np.zeros((args.num_envs, action_dim), dtype=np.float32)

    num_bump = 0
    while (True in has_bump) or (True in has_bump_dvt):
        num_bump += 1
        assert num_bump < 500 , "policy with bad local optimum"

        if with_discoveryPi:
            "update policy distribution and sample action"
            if args.resetPi:
                TensActions[has_bump], Tens_logPi[has_bump], TensActions_dvt[has_bump_dvt], Tens_logPi_dvt[
                    has_bump_dvt], log_sig[has_bump], mu[has_bump], LogPi_jacobian[has_bump] = \
                    policy_last_layer(np2torch(states[:args.nEnv_perPi][has_bump]), pi_head, mu_tail, log_sig_tail,
                                      s_dvt=np2torch(states[args.nEnv_perPi:][has_bump_dvt]), pi_head_dvt=pi_head_dvt,
                                      mu_tail_dvt=mu_tail_dvt,
                                      log_sig_tail_dvt=log_sig_tail_dvt, config=config,
                                      save_pi_logs=True)
                ActEnv_dvt = pytorch2numpy(TensActions_dvt)
            else:
                TensActions[has_bump], Tens_logPi[has_bump], log_sig[has_bump], mu[has_bump], LogPi_jacobian[has_bump] = \
                    policy_last_layer(np2torch(states[has_bump]), pi_head, mu_tail, log_sig_tail, config=config,
                                      save_pi_logs=True)
            ActEnv = pytorch2numpy(TensActions)
        else:
            ActEnv[has_bump] = np.random.uniform(low=-1, high=1, size=(sum(has_bump), action_dim))
        if args.bumpDetection:
            has_bump = np.array([env_.bump_detection(a_) for env_, a_ in zip(envs[:args.nEnv_perPi], ActEnv)])
            if args.resetPi:
                has_bump_dvt = np.array(
                    [env_.bump_detection(a_) for env_, a_ in zip(envs[args.nEnv_perPi:], ActEnv_dvt)])
        else:
            has_bump = np.zeros((args.nEnv_perPi), dtype=np.bool)
            has_bump_dvt = np.zeros((args.nEnv_perPi), dtype=np.bool)

    if with_discoveryPi:
        if args.resetPi:
            ActEnv = pytorch2numpy(torch.cat((TensActions, TensActions_dvt), dim=0))
        else:
            ActEnv = pytorch2numpy(TensActions)
        DetLogCov_sum += pytorch2numpy(sumFct(2 * log_sig))  # compute determinant of Log covariance
        CovNorm_sum += pytorch2numpy(meanFctNorm((2 * log_sig).exp()))  # compute norm of covariance
        muNorm_sum += pytorch2numpy(meanFctNorm(mu))
        LogPi_jacobian_sum += pytorch2numpy(LogPi_jacobian).mean()
    else:
        TensActions = numpy2pytorch(ActEnv, differentiable=False, device=device)


    "Make a step"
    for step_rep in range(actionRepeat):
        obs, _, dones, _ = zip(*[env_.step(ActEnv[i_]) for i_, env_ in enumerate(envs)])
        if config['n_stack'] > 1:
            if (step_rep + 1) > (config['actionRepeat'] - config['n_stack']):
                next_observations[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = obs
        elif (step_rep + 1) == actionRepeat:
            assert step_rep < 2, 'actionRepeat is already performed in env'
            next_observations = np.vstack(obs)

    if args.evalExplor:
        robot_pos[(elapsed_steps - 1) % (args.backprop_per_eval)] = np.array(
            [env_.object.copy() for env_ in envs])
        endDetections = [env_.endDetector(debug=args.debug) for env_ in envs]
        if True in endDetections:
            stop_training = True
            endDetection_dict['endDetection'] = 'epochs {}, gradients {}, pi-gradients {}, steps {}'.format(
                elapsed_epochs, elapsed_gradients, pi_elapsed_gradients, elapsed_steps)
            endDetection_dict['epochs'] = elapsed_epochs
            endDetection_dict['gradients'] = elapsed_gradients
            endDetection_dict['steps'] = elapsed_steps
            print('\n  robot touch the end at step {}'.format(elapsed_steps))

            saveJson(endDetection_dict, save_path, 'endDetection.json')

    "update reset info"
    envs_to_reset = np.array(dones, dtype=np.bool)
    reseted = np.arange(args.num_envs)[envs_to_reset]

    "Compute next state"
    o_alpha = alpha(np2torchDev(observations))
    o_beta = beta(torch.cat((np2torchDev(states), np2torchDev(ActEnv)), dim=1))
    input_gamma = torch.cat((o_alpha, o_beta), dim=1)
    s_next = gamma(input_gamma)

    "update SRL network"
    "Reconstruct observations of current step for all trajectories"
    xHat_next = omega_last_layer(omega(s_next))
    lossNextObs = (Loss_obs(xHat_next, np2torchDev(next_observations), args.num_envs).sum(
        (1, 2, 3))).sum()
    "update alpha/beta/gamma networks"
    optimizer.zero_grad()
    lossNextObs.backward()
    optimizer.step()
    lossNextObs_log[k_srl] = pytorch2numpy(lossNextObs)

    if args.resetPi:
        lossNextObs_dvt1_sum += pytorch2numpy((Loss_obs(xHat_next.to('cpu').detach()[:args.nEnv_perPi],
                                                        np2torch(next_observations[:args.nEnv_perPi]),
                                                        args.nEnv_perPi).sum((1, 2, 3))).sum())
        lossNextObs_dvt2_sum += pytorch2numpy((Loss_obs(xHat_next.to('cpu').detach()[args.nEnv_perPi:],
                                                        np2torch(next_observations[args.nEnv_perPi:]),
                                                        args.nEnv_perPi).sum((1, 2, 3))).sum())

    elapsed_gradients += 1
    k_srl += 1

    s_next = pytorch2numpy(s_next)
    "update policy rewards"
    if with_discoveryPi and (elapsed_steps % args.pi_sampling == 0):
        if args.inverse:
            actions_hat = iota(torch.cat((np2torch(states), np2torch(s_next)), dim=1))
            lossInverseSequential += (Loss(actions_hat, np2torch(ActEnv),args.num_envs).sum(-1)).sum()
            r_inverseSequential += (Loss(actions_hat[:args.nEnv_perPi].detach(), TensActions, args.nEnv_perPi).sum(-1)).sum()

            if args.resetPi:
                r_inverseSequential_dvt += (Loss(actions_hat[args.nEnv_perPi:].detach(), TensActions_dvt,args.nEnv_perPi).sum(-1)).sum()

        if args.entropy:
            r_entropySequential -= Tens_logPi.mean()
            if args.resetPi: r_entropySequential_dvt -= Tens_logPi_dvt.mean()
        else:
            r_entropySequential -= Tens_logPi.mean().detach()

        if args.LPB:
            "Compute state with current networks and keep action gradients for LPB backprop"
            o_alphaLPB = alpha(np2torchDev(observations[:args.nEnv_perPi])).to('cpu').detach()
            if device == 'cuda': beta.to('cpu'), gamma.to('cpu')
            o_betaLPB = beta(
                torch.cat((np2torch(states[:args.nEnv_perPi]), TensActions),
                          dim=1))  # keep only dependance on action parameters
            input_gammaLPB = torch.cat((o_alphaLPB, o_betaLPB), dim=1)
            s_nextLPB = gamma(input_gammaLPB)

            "Compute state with target networks without action gradients, detach() is not needed as requires_grad=False"
            if device == 'cuda': alpha_old.to(device)
            o_alpha_old = alpha_old(np2torchDev(observations[:args.nEnv_perPi])).to('cpu')
            o_beta_old = beta_old(
                torch.cat((np2torch(states[:args.nEnv_perPi]), TensActions.detach()), dim=1))
            input_gamma_old = torch.cat((o_alpha_old, o_beta_old), dim=1)
            s_next_old = gamma_old(input_gamma_old)
            # to be maximized by policy
            r_LPBsequential += (Loss(s_next_old, s_nextLPB, args.nEnv_perPi).sum(-1)).sum()

            if args.resetPi:
                o_alphaLPB_dvt = alpha(np2torchDev(observations[args.nEnv_perPi:])).to('cpu').detach()
                o_betaLPB_dvt = beta(torch.cat((np2torch(states[args.nEnv_perPi:]), TensActions_dvt), dim=1))
                input_gammaLPB_dvt = torch.cat((o_alphaLPB_dvt, o_betaLPB_dvt), dim=1)
                s_nextLPB_dvt = gamma(input_gammaLPB_dvt)
                ""
                o_alpha_old_dvt = alpha_old(np2torchDev(observations[args.nEnv_perPi:])).to('cpu')
                o_beta_old_dvt = beta_old(
                    torch.cat((np2torch(states[args.nEnv_perPi:]), TensActions_dvt.detach()), dim=1))
                input_gamma_old_dvt = torch.cat((o_alpha_old_dvt, o_beta_old_dvt), dim=1)
                s_next_old_dvt = gamma_old(input_gamma_old_dvt)
                r_LPBsequential_dvt += (Loss(s_next_old_dvt, s_nextLPB_dvt,args.nEnv_perPi).sum(-1)).sum()

            if device == 'cuda':
                alpha_old.to('cpu'), beta.to(device), gamma.to(device)

    "update observations/states"
    observations = add_noise(next_observations.copy(), noise_adder, config)
    states = s_next

    if True in dones:
        "update reset observations/states"
        assert args.maxStep != np.inf, 'maxStep == np.inf !'
        obs = np.vstack([envs[i_].reset() for i_ in np.arange(args.num_envs)[envs_to_reset]])
        if config['n_stack'] > 1:
            obs = reset_stack(obs, config)
        resetStates = resetState(obs, alpha, beta, gamma, config)
        states[reseted] = resetStates
        observations[reseted] = obs

    if with_discoveryPi and (elapsed_steps % args.pi_updateInt == 0):
        "save entropy_log"
        entropy_log[k_pi] = pytorch2numpy(r_entropySequential) / args.nPi_samples
        if args.resetPi and not (args.weightInverse > 0 or args.weightLPB > 0):
            entropy_log_dvt[k_pi] = pytorch2numpy(r_entropySequential_dvt) / args.nPi_samples

        if args.inverse:
            "update inverse model"
            iotaOptimizer.zero_grad()
            lossInverseSequential.backward()
            iotaOptimizer.step()
            "plot the inverse rewards used for discoveryPi updates"
            lossInverse_log[k_pi] = pytorch2numpy(r_inverseSequential / args.nPi_samples)
            if args.resetPi: lossInverse_log_dvt[k_pi] = pytorch2numpy(r_inverseSequential_dvt / args.nPi_samples)

        if args.autoEntropyTuning:
            temperature = pytorch2numpy(temperatureTensor).item()
            temperature_log[k_pi] = temperature

            log_temperature_optim.zero_grad()
            temperature_loss = temperatureTensor * (r_entropySequential.detach() - args.nPi_samples * target_entropy)
            temperature_loss.backward()
            log_temperature_optim.step()
            "update temperature for next policy update"
            temperatureTensor = log_temperatureTensor.exp()
        else:
            temperature = args.weightEntropy

        if args.LPB:
            "backward inverse error/LPB gradients to the policy"
            LPB_log[k_pi] = pytorch2numpy(r_LPBsequential / args.nPi_samples)
            if args.resetPi: LPB_log_dvt[k_pi] = pytorch2numpy(r_LPBsequential_dvt / args.nPi_samples)

            "Update old network with current network's weights"
            alpha_old = update_target_network(alpha_old, alpha, device=device)
            if device == 'cuda':
                beta.to('cpu'), gamma.to('cpu')
            beta_old = update_target_network(beta_old, beta)
            gamma_old = update_target_network(gamma_old, gamma)

        "update policy"
        policyOptimizer.zero_grad()
        lossPi = - args.weightInverse * r_inverseSequential - args.weightLPB * r_LPBsequential - temperature * r_entropySequential
        lossPi.backward()
        policyOptimizer.step()

        DetLogCov_log[k_pi] = DetLogCov_sum / args.pi_updateInt
        CovNorm_log[k_pi] = CovNorm_sum / args.pi_updateInt
        muNorm_log[k_pi] = muNorm_sum / args.pi_updateInt
        LogPi_jacobian_log[k_pi] = LogPi_jacobian_sum / args.pi_updateInt
        k_pi += 1
        pi_elapsed_gradients += 1
        "reset resetPi sequential-rewards"
        lossInverseSequential = r_entropySequential = r_inverseSequential = r_LPBsequential = 0
        DetLogCov_sum = CovNorm_sum = muNorm_sum = LogPi_jacobian_sum = 0

        if args.resetPi:
            "update resetPi policy"
            policyOptimizer_dvt.zero_grad()
            lossPi_dvt = - args.weightInverse * r_inverseSequential_dvt - args.weightLPB * r_LPBsequential_dvt - temperature * r_entropySequential_dvt
            lossPi_dvt.backward()
            policyOptimizer_dvt.step()
            "reset resetPi sequential-rewards"
            r_inverseSequential_dvt = r_entropySequential_dvt = r_LPBsequential_dvt = 0

        if args.LPB and device == 'cuda':
            beta.to(device), gamma.to(device)

    if elapsed_gradients > 1 and (k_srl == args.backprop_per_eval) or stop_training:

        "Plot losses"
        config['elapsed_steps'] = elapsed_steps
        config['elapsed_gradients'] = elapsed_gradients
        config['elapsed_epochs'] = elapsed_epochs = elapsed_gradients // args.backprop_per_eval
        config['elapsed_time_s'] = time() - start_time + loaded_time_s
        config['elapsed_time'] = elapsed_time = str(timedelta(seconds=(int(config['elapsed_time_s']))))

        # Plot loss
        update_text(lossNextObs_log, save_path + "lossNextObs_log.txt")
        lossNextObs_mean = np.mean(lossNextObs_log)
        config['lossNextObs_mean'] = float2string(lossNextObs_mean)
        update_text([lossNextObs_mean], save_path + "lossNextObs_mean.txt")
        plotter(lossNextObs_mean, save_path, name='lossNextObs_mean', title="{} \n{}".format(SRL_name, hashCode),
                ylabel='Average next obs prediction errors',  # over an epoch of %s updates' % args.backprop_per_eval,
                backprop_per_eval=args.backprop_per_eval, text_file=save_path + "lossNextObs_mean.txt")

        "Real validation"
        with open(save_path + "lossNextObs_mean.txt") as f:
            loss_log = np.array(f.read().split("\n"), dtype=np.float32)
            lossNextObs_avg = np.mean(loss_log[-args.patience:])

        print(
            "{} \n[{}]  ~~~~~~ Training only {} {} ~~~~~~\n[{}] Epoch: {} | Gradient steps: {} | {}: {:.3f} | {}: {:.3f}".format(
                env_params_name, hashCode, args.method, SRL_name, elapsed_time, elapsed_epochs, elapsed_gradients,
                'lossNextObs_mean', lossNextObs_mean, 'lossNextObs_avg', lossNextObs_avg))
        if all_dir:
            print('  all_dir: {}'.format(config['all_dir']))

        early_stopper(-lossNextObs_avg, Nepochs=elapsed_epochs)
        suffix = 'best' if early_stopper.counter == 0 else 'last'

        if args.resetPi:
            "dvt1"
            lossNextObs_dvt1_mean = lossNextObs_dvt1_sum / args.backprop_per_eval
            config['lossNextObs_dvt1_mean'] = float2string(lossNextObs_dvt1_mean)
            update_text([lossNextObs_dvt1_mean], save_path + "lossNextObs_dvt1_mean.txt")
            plotter(lossNextObs_dvt1_mean, save_path, name='lossNextObs_dvt1_mean',
                    title="{} \n{}".format(SRL_name, hashCode),
                    ylabel='Average next obs prediction errors',
                    backprop_per_eval=args.backprop_per_eval, text_file=save_path + "lossNextObs_dvt1_mean.txt")
            "dvt2"
            lossNextObs_dvt2_mean = lossNextObs_dvt2_sum / args.backprop_per_eval
            config['lossNextObs_dvt2_mean'] = float2string(lossNextObs_dvt2_mean)
            update_text([lossNextObs_dvt2_mean], save_path + "lossNextObs_dvt2_mean.txt")
            plotter(lossNextObs_dvt2_mean, save_path, name='lossNextObs_dvt2_mean',
                    title="{} \n{}".format(SRL_name, hashCode),
                    ylabel='Average next obs prediction errors',
                    backprop_per_eval=args.backprop_per_eval, text_file=save_path + "lossNextObs_dvt2_mean.txt")
            "compute mean intrinsic rewards of both policies"
            if args.weightInverse > 0 or args.weightLPB > 0:
                reward_dvt1_mean = args.weightInverse * np.mean(lossInverse_log) + args.weightLPB * np.mean(LPB_log)
                reward_dvt2_mean = args.weightInverse * np.mean(lossInverse_log_dvt) + args.weightLPB * np.mean(
                    LPB_log_dvt)
            elif args.weightEntropy > 0:
                reward_dvt1_mean = np.mean(entropy_log)
                reward_dvt2_mean = np.mean(entropy_log_dvt)

            config['reward_dvt1_mean'] = float2string(reward_dvt1_mean)
            config['reward_dvt2_mean'] = float2string(reward_dvt2_mean)
            update_text([reward_dvt1_mean], save_path + "reward_dvt1_mean.txt")
            update_text([reward_dvt2_mean], save_path + "reward_dvt2_mean.txt")
            plotter(reward_dvt1_mean, save_path, name='reward_dvt1_mean',
                    title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel='Average intrinsic reward 1st policy',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "reward_dvt1_mean.txt")
            plotter(reward_dvt2_mean, save_path, name='reward_dvt2_mean',
                    title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel='Average intrinsic reward 2nd policy',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "reward_dvt2_mean.txt")
            print(' ' * 100 + 'reward_dvt1_mean: {:.3f}           reward_dvt2_mean: {:.3f}\n'.format(reward_dvt1_mean,
                                                                                                     reward_dvt2_mean) +
                  ' ' * 100 + 'lossNextObs_dvt1_mean: {:.3f}    lossNextObs_dvt2_mean: {:.3f}'.format(
                lossNextObs_dvt1_mean, lossNextObs_dvt2_mean))

            "early stop 2nd discovery policy"
            score_dvt1 = f_stopper(reward_dvt1_mean)
            score_dvt2 = f_stopper(reward_dvt2_mean)
            if score_dvt2 > reward_dvt1_mean:
                dvt_counter += 1
                dvt_before2reset = dvtEquiv_before2reset = 0
                print(' ' * 100 + f'dvt_counter: {dvt_counter} out of {args.dvt_patience}')
            elif score_dvt1 > reward_dvt2_mean:
                dvt_before2reset += 1
                dvt_counter = dvtEquiv_before2reset = 0
                print(' ' * 100 + f'dvt_before2reset: {dvt_before2reset} out of {args.dvt_patience}')
            elif not keep1stPolicy:
                dvtEquiv_before2reset += 1
                dvt_counter = dvt_before2reset = 0
                print(' ' * 100 + f'dvtEquiv_before2reset: {dvtEquiv_before2reset} out of {args.dvt_patience}')
            elif keep1stPolicy:
                dvt_counter = dvt_before2reset = 0
                print(' ' * 100 + f'dvtEquiv and keep1stPolicy')
            if (dvt_counter >= args.dvt_patience) or (dvt_before2reset >= args.dvt_patience) or (
                    dvtEquiv_before2reset >= args.dvt_patience):
                if dvt_counter >= args.dvt_patience:
                    "change 1st policy by the 2nd one which is better"
                    print(' ' * 100 + f'~~ follow 2nd policy and start new one ~~')
                    keep1stPolicy_counter = 0
                    "if the 2nd policy becomes better then better to follow the 2nd policy and reset the 1st policy" \
                    "which is equivalent to update the 1st policy with 2nd policy, and reset the 2nd policy"
                    keep1stPolicy = False

                    pi_head = update_target_network(pi_head, pi_head_dvt)
                    mu_tail = update_target_network(mu_tail, mu_tail_dvt)
                    log_sig_tail = update_target_network(log_sig_tail, log_sig_tail_dvt)
                    # pi_head.to(cpu).train(), mu_tail.to(cpu).train(), log_sig_tail.to(cpu).train()
                    policyOptimizer = torch.optim.Adam(
                        list(pi_head.parameters()) + list(mu_tail.parameters()) + list(log_sig_tail.parameters()),
                        lr=args.lr_explor)
                    update_text('[epoch {}] dvt_counter >= {}'.format(elapsed_epochs, args.dvt_patience),
                                save_path + "resetPi_log.txt", text=True)
                elif dvt_before2reset >= args.dvt_patience:
                    keep1stPolicy = False
                    print(' ' * 100 + f'~~ reset 2nd policy as it is not better than 1st one ~~')
                    keep1stPolicy_counter += 1
                    update_text('[epoch {}] dvt_before2reset >= {}'.format(elapsed_epochs, args.dvt_patience),
                                save_path + "resetPi_log.txt", text=True)
                    if keep1stPolicy_counter >= 5:
                        "when this happen 5 times then better to reset 2nd policy with 1st one"
                        keep1stPolicy = True
                        keep1stPolicy_counter = 0
                        print(' ' * 100 + f'~~~~~ keep1stPolicy as it is better ~~~~~')
                        update_text('keep1stPolicy_counter >= 5', save_path + "resetPi_log.txt", text=True)
                elif dvtEquiv_before2reset >= args.dvt_patience:
                    "cannot happen when keep1stPolicy==True"
                    print(' ' * 100 + f'~~ reset 2nd policy as it is equivalent to 1st one ~~')
                    keep1stPolicy_counter = 0
                    update_text('[epoch {}] dvtEquiv_before2reset >= {}'.format(elapsed_epochs, args.dvt_patience),
                                save_path + "resetPi_log.txt", text=True)
                dvt_counter = dvt_before2reset = dvtEquiv_before2reset = 0

                if keep1stPolicy:
                    "update the 2nd policy with 1st policy"
                    pi_head_dvt = update_target_network(pi_head_dvt, pi_head)
                    mu_tail_dvt = update_target_network(mu_tail_dvt, mu_tail)
                    log_sig_tail_dvt = update_target_network(log_sig_tail_dvt, log_sig_tail)
                else:
                    "reset the 2nd policy"
                    pi_head_dvt, mu_tail_dvt, log_sig_tail_dvt, _ = MLP_mdn(args.state_dim,
                                                                            get_hidden(args.nb_hidden_pi) + [
                                                                                action_dim],
                                                                            cutoff=args.cutoff,
                                                                            activation=args.activation)
                pi_head_dvt.to(cpu).train(), mu_tail_dvt.to(cpu).train(), log_sig_tail_dvt.to(cpu).train()
                policyOptimizer_dvt = torch.optim.Adam(
                    list(pi_head_dvt.parameters()) + list(mu_tail_dvt.parameters()) + list(
                        log_sig_tail_dvt.parameters()), lr=args.lr_explor)

        print('  Saving models ......')
        "Save models"
        alpha.eval(), beta.eval(), gamma.eval(), omega.eval()
        save_model([alpha, beta, gamma], save_path + 'state_model')
        save_model(omega, save_path + 'state_model_tail')
        if with_discoveryPi:
            config['pi_elapsed_gradients'] = pi_elapsed_gradients
            pi_head.eval(), mu_tail.eval(), log_sig_tail.eval()
            save_model([pi_head, mu_tail, log_sig_tail], save_path + 'piExplore')
            if args.inverse:
                save_model(iota, save_path + 'iota')

        "do not depend on args.nPi_samples: loss_log DetLogCov_log, CovNorm_log, muNorm_log, LogPi_jacobian_log"
        if with_discoveryPi:
            update_text(DetLogCov_log, save_path + "DetLogCov_log.txt")
            DetLogCov_mean = np.mean(DetLogCov_log)
            config['DetLogCov_mean'] = float2string(DetLogCov_mean)
            update_text([DetLogCov_mean], save_path + "DetLogCov_mean.txt")
            plotter(DetLogCov_mean, save_path, name='DetLogCov_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel=r'Average $det(log (\Sigma))$',
                    # avg over %s policy updates' % args.pi_backprop_per_eval,
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "DetLogCov_mean.txt")
            update_text(CovNorm_log, save_path + "CovNorm_log.txt")
            CovNorm_mean = np.mean(CovNorm_log)
            config['CovNorm_mean'] = float2string(CovNorm_mean)
            update_text([CovNorm_mean], save_path + "CovNorm_mean.txt")
            plotter(CovNorm_mean, save_path, name='CovNorm_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel=r'Average $\Vert\Sigma\Vert_2$',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "CovNorm_mean.txt")

            update_text(muNorm_log, save_path + "muNorm_log.txt")
            muNorm_mean = np.mean(muNorm_log)
            config['muNorm_mean'] = float2string(muNorm_mean)
            update_text([muNorm_mean], save_path + "muNorm_mean.txt")
            plotter(muNorm_mean, save_path, name='muNorm_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel=r'Average $\Vert\mu_{\pi}(\mathbf{s}_{t})\Vert_2$',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "muNorm_mean.txt")

            update_text(LogPi_jacobian_log, save_path + "LogPi_jacobian_log.txt")
            LogPi_jacobian_mean = np.mean(LogPi_jacobian_log)
            config['LogPi_jacobian_mean'] = float2string(LogPi_jacobian_mean)
            update_text([LogPi_jacobian_mean], save_path + "LogPi_jacobian_mean.txt")
            plotter(LogPi_jacobian_mean, save_path, name='LogPi_jacobian_mean',
                    title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel=r'Average Jacobian of $log(\pi(\cdot\vert\mathbf{s}_{t})$)',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "LogPi_jacobian_mean.txt")

            update_text(entropy_log, save_path + "entropy_log.txt")
            entropy_mean = np.mean(entropy_log)
            config['entropy_mean'] = float2string(entropy_mean)
            update_text([entropy_mean], save_path + "entropy_mean.txt")
            plotter(entropy_mean, save_path, name='entropy_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel=r'Average entropy of $\pi(\cdot \vert \mathbf{s}_{t})$',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "entropy_mean.txt")

            print('  DetLogCov: {:.3f}  entropy_mean: {:.3f}'.format(DetLogCov_mean, entropy_mean))
        if args.inverse:
            update_text(lossInverse_log, save_path + "lossInverse_log.txt")
            lossInverse_mean = np.mean(lossInverse_log)
            config['lossInverse_mean'] = float2string(lossInverse_mean)
            update_text([lossInverse_mean], save_path + "lossInverse_mean.txt")
            plotter(lossInverse_mean, save_path, name='lossInverse_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel='Average inverse prediction errors',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "lossInverse_mean.txt")
        if args.LPB:
            update_text(LPB_log, save_path + "LPB_log.txt")
            LPB_mean = np.mean(LPB_log)
            config['LPB_mean'] = float2string(LPB_mean)
            update_text([LPB_mean], save_path + "LPB_mean.txt")
            plotter(LPB_mean, save_path, name='LPB_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel='Average LPB',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "LPB_mean.txt")

        if args.autoEntropyTuning:
            update_text(temperature_log, save_path + "temperature_log.txt")
            temperature_mean = np.mean(temperature_log)
            config['temperature_mean'] = float2string(temperature_mean, nfloat=4)
            update_text([temperature_mean], save_path + "temperature_mean.txt")
            plotter(temperature_mean, save_path, name='temperature_mean', title="{} \n{}".format(SRL_name, hashCode),
                    xlabel='policy gradient steps', ylabel='Average temperature',
                    backprop_per_eval=args.pi_backprop_per_eval, text_file=save_path + "temperature_mean.txt")

        if args.evalExplor:
            update_text(robot_pos, save_path_robotPos + "robotPos_E{:06d}".format(elapsed_epochs), array=True)

        "get exploration visualization with visuExplor_env with high maxSteps"
        # if not args.debug:
        nextObsEval = XSRL_nextObsEval(alpha, beta, gamma, omega, config, save_path, gradientStep=elapsed_gradients, saved_step=(elapsed_epochs - 1), suffix=suffix, debug=args.debug)

        config['nextObsEval'] = float2string(nextObsEval)
        update_text([nextObsEval], save_path + "nextObsEval_log.txt")
        plotter(nextObsEval, save_path, name='nextObsEval', title="{} \n{}".format(SRL_name, hashCode),
                ylabel=r'Average next-obs prediction errors at test-time (after %s updates)' % args.backprop_per_eval,
                backprop_per_eval=args.backprop_per_eval, text_file=save_path + "nextObsEval_log.txt")
        piExplore2obs(visuExplor_env, noise_adder, alpha, beta, gamma, omega, pi_head, mu_tail, log_sig_tail,
                      config, save_path, suffix=suffix, debug=args.debug)
        if args.env_name in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:
            getPiExplore(visuExplor_env, noise_adder, alpha, beta, gamma, pi_head, mu_tail, log_sig_tail,
                         config, visuExplor_path, elapsed_epochs,
                         debug=args.debug)

        "Save config.pkl and config.json"
        saveConfig(config, save_dir=save_path)
        saveJson(config, save_path)

        "re-init logs"
        k_pi = k_srl = 0
        lossNextObs_log = np.zeros((args.backprop_per_eval), np.float32)
        if with_discoveryPi:
            DetLogCov_log, CovNorm_log, muNorm_log, LogPi_jacobian_log = [
                np.zeros((args.pi_backprop_per_eval), np.float32) for _ in
                range(4)]
            lossInverse_log, LPB_log, entropy_log, temperature_log = [np.zeros(
                (args.pi_backprop_per_eval)) for _ in range(4)]
        if args.resetPi:
            lossNextObs_dvt1_sum = lossNextObs_dvt2_sum = 0
            if args.weightInverse > 0 or args.weightLPB > 0:
                lossInverse_log_dvt, LPB_log_dvt = [np.zeros((args.pi_backprop_per_eval), np.float32) for _ in range(2)]
            elif args.weightEntropy > 0:
                entropy_log_dvt = np.zeros((args.pi_backprop_per_eval), np.float32)
        if args.evalExplor:
            robot_pos = np.zeros((args.backprop_per_eval, args.num_envs, 2), np.float32)

        "Evaluation loop"
        print('  EvalAgent is predicting next obs ......')
        images_hat, images, im_next, im_high_render = np.zeros([evalSteps, *image_size]), np.zeros(
            [evalSteps, *image_size]), np.zeros([evalSteps, *image_size]), np.zeros(
            [evalSteps, 256, 256, 3])
        for evalStep in range(evalSteps):
            with torch.no_grad():  # detach() not needed when torch.no_grad
                "Do one step with every robots"
                "Make a step"
                if with_discoveryPi:
                    "update policy distribution and sample action"
                    TensA = policy_last_layer(np2torch(evalState), pi_head, mu_tail, log_sig_tail,
                                              config=config).to(device)
                    ActEnv = pytorch2numpy(TensA.squeeze(dim=0))
                else:
                    ActEnv = envEval.action_space.sample()
                    TensA = numpy2pytorch(ActEnv, differentiable=False, device=device).unsqueeze(dim=0)
                "Make an evaluation step"
                for step_rep in range(actionRepeat):
                    obs, _, done, _ = envEval.step(ActEnv)
                    if config['n_stack'] > 1:
                        if (step_rep + 1) > (config['actionRepeat'] - config['n_stack']):
                            next_observation[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = obs
                    elif (step_rep + 1) == actionRepeat:
                        assert step_rep < 2, 'actionRepeat is already performed in env'
                        next_observation = obs

                if done:
                    observation = envEval.reset()
                    if config['n_stack'] > 1:
                        observation = reset_stack(observation, config)
                    evalState = resetState(observation, alpha, beta, gamma, config)
                    "save reset"
                    images_hat[evalStep] = np.zeros((image_size), np.float32)
                    images[evalStep] = np.zeros((image_size), np.float32)
                    im_next[evalStep] = np.zeros((image_size), np.float32)
                    im_high_render[evalStep] = np.zeros((256, 256, 3), np.float32)
                    continue

                "Predict next states"
                o_alpha = alpha(np2torchDev(observation))
                o_beta = beta(torch.cat((np2torchDev(evalState), TensA), dim=1))
                input_gamma = torch.cat((o_alpha, o_beta), dim=1)
                s_nextEval = gamma(input_gamma)

                "Reconstruct observations of current step for all trajectories"
                xHat_nextEval = pytorch2numpy(omega_last_layer(omega(s_nextEval)))
                "save outputs"
                images_hat[evalStep] = xHat_nextEval[:, -3:, :, :]
                images[evalStep] = observation[:, -3:, :, :]
                im_next[evalStep] = next_observation[:, -3:, :, :]
                im_high_render[evalStep] = render_env(envEval, 256, False, camera_id_eval, True,
                                                      downscaling=False) / 255.
                "update inputs"
                evalState = pytorch2numpy(s_nextEval)
                observation = add_noise(next_observation.copy(), noise_adder, config)

        "Reconstruct observations for visualization"
        frame = 0
        frames = slice(frame, frame + evalSteps)
        "reshape to HWC"
        images_hat = images_hat[frames].transpose(0, 2, 3, 1)
        images = images[frames].transpose(0, 2, 3, 1)
        im_next = im_next[frames].transpose(0, 2, 3, 1)

        "save last prediction"
        last_idx = -1 if not done else -2
        plot_xHat(images[last_idx], images_hat[last_idx], imgTarget=im_next[last_idx],
                  im_high_render=im_high_render[last_idx], imLabel=imLabel,
                  figure_path=save_path, with_noise=with_noise, with_nextObs=True,
                  gradientStep=elapsed_gradients, suffix=suffix)
        "Reconstruct observations for visualization"
        for i in range(len(images)):
            plot_xHat(images[i], images_hat[i], imgTarget=im_next[i], im_high_render=im_high_render[i], imLabel=imLabel,
                      figure_path=save_path + 'xHat', with_noise=with_noise, with_nextObs=True,
                      gradientStep=elapsed_gradients,
                      saved_step=(i + evalSteps * (elapsed_epochs - 1)))

        print('  end XSRL evaluation')
        alpha.train(), beta.train(), gamma.train(), omega.train()
        if with_discoveryPi:
            pi_head.train(), mu_tail.train(), log_sig_tail.train()
        "force the Garbage Collector to release unreferenced memory"
        del images, im_next, images_hat, im_high_render
        gc.collect()

if early_stopper.early_stop:
    config['early_stop'] = True
    saveConfig(config, save_dir=save_path)
    saveJson(config, save_path)
