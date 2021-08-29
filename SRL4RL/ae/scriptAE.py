"""
23/03/2020
Astrid Merckling
AE/RAE/VAE
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

from SRL4RL.utils.nn_torch import CNN, CNN_Transpose, MLP_Module, Flatten, pytorch2numpy, numpy2pytorch, layers_MLP, save_model, set_seeds
from SRL4RL.utils.utils import createFolder, saveConfig, EarlyStopping, loadConfig, update_text, float2string, saveJson, giveSRL_name
from SRL4RL.utils.env_wrappers import BulletWrapper
from SRL4RL import SRL4RL_path

from SRL4RL.utils.utilsPlot import plot_xHat, plotter
from SRL4RL.utils.utilsEnv import add_noise, render_env, giveEnv_name, update_video, reset_stack
from SRL4RL.ae.utils import AE_nextObsEval, decoder_last_layer, reparametrize
from SRL4RL.ae.arguments import get_args, update_args_AE, assert_args_AE
from SRL4RL.rl.modules.agent_utils import loadPi, process_inputs_gt

"register bullet_envs in gym"
import bullet_envs.__init__

# get the params
args = get_args()
args = update_args_AE(args)

"Pretraining setup"
all_dir = None
if args.dir:
    if args.dir[-1] != '/': args.dir += '/'
    loaded_config = loadConfig(args.dir)
    dir_hashCode = loaded_config['hashCode']
    remove_keys = ['keep_seed', 'dir', 'debug','patience', 'n_epochs']
    if args.keep_seed:
        assert loaded_config['hashCode'] == dir_hashCode
        if 'all_dir' in loaded_config:
            all_dir = loaded_config['all_dir']
    else:
        remove_keys += ['logs_dir', 'seed', 'noise_type','lr', 'patience']
        if 'all_dir' not in loaded_config:
            all_dir = dir_hashCode
        else:
            all_dir = loaded_config['all_dir'] + '-' + dir_hashCode
 
    if args.keep_seed:
        select_new_args = {k: args.__dict__[k] for k in remove_keys}
        loaded_config.update(select_new_args)
    else:
        keep_keys = list(args.__dict__.keys())
        print(keep_keys)
        for k in remove_keys:
            keep_keys.remove(k)
        keep_keys += ['n_stack']
        select_old_args = {k: loaded_config[k] for k in keep_keys}
        args.__dict__.update(select_old_args)           

    "force the Garbage Collector to release unreferenced memory"
    del select_new_args, keep_keys
    gc.collect()

maxStep_eval = 500
if args.debug:
    "We use num_envs parallel envs for data collection while training"
    args.num_envs = 4
    args.backprop_per_eval = 64
    # args.maxStep = 30
else:
    matplotlib.use('Agg')

assert_args_AE(args)

"Training settings"
cpu = torch.device('cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
args.n_epochs = int(args.n_epochs)

np2torch = lambda x: numpy2pytorch(x, differentiable=False, device=device)


if args.randomExplor:
    if 'Turtlebot' in args.env_name:
        args.maxStep = 50
    elif args.env_name == 'ReacherBulletEnv-v0':
        args.maxStep = 50
    elif args.env_name == 'InvertedPendulumSwingupBulletEnv-v0':
        args.maxStep = 250
    elif args.env_name == 'HalfCheetahBulletEnv-v0':
        args.maxStep = 250

if args.maxStep == np.inf:
    maxSteps = [args.maxStep] * args.num_envs
else:
    "to remove correlations between each robot trajectory"
    maxSteps = np.linspace(args.maxStep, args.maxStep + args.num_envs - 1, args.num_envs).astype(int)
    print('maxSteps: ', maxSteps)

"Environment settings"
evalSteps = 5  # to evaluate reconstructed images
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
    envs = [gym.make(args.env_name, renders=False, maxSteps=maxSteps[i_], actionRepeat=args.actionRepeat,
                     image_size=image_size[-1], color=True, seed=args.seed + i_ + 1,
                     with_target=False, noise_type='none', fpv=args.fpv, display_target=False,
                     randomExplor=args.randomExplor,wallDistractor=args.wallDistractor,
                     distractor=args.distractor) for i_ in range(args.num_envs)]

    envEval = gym.make(args.env_name, renders=False, maxSteps=maxStep_eval, actionRepeat=args.actionRepeat,
                       image_size=image_size[-1], color=True, seed=args.seed,
                       with_target=False, noise_type='none', fpv=args.fpv, display_target=False,
                       randomExplor=args.randomExplor,wallDistractor=args.wallDistractor,
                       distractor=args.distractor)

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
    envs = [gym.make('PybulletEnv-v0', env_name=args.env_name, renders=False, maxSteps=maxSteps[i_] * args.actionRepeat,
                     actionRepeat=actionRepeat_env,
                     image_size=image_size[-1], color=True, seed=args.seed + i_ + 1,
                     noise_type='none', fpv=args.fpv, doneAlive=False, randomExplor=args.randomExplor,
                     distractor=args.distractor) for i_ in
            range(args.num_envs)]
    envEval = gym.make('PybulletEnv-v0', env_name=args.env_name, renders=False,
                       maxSteps=maxStep_eval * args.actionRepeat, actionRepeat=actionRepeat_env,
                       image_size=image_size[-1], color=True, seed=args.seed,
                       noise_type='none', fpv=args.fpv, doneAlive=False, randomExplor=args.randomExplor,
                       distractor=args.distractor)

[env_.seed(args.seed + i_ + 1) for i_, env_ in enumerate(envs)]
envEval.seed(args.seed)

"wrap envs"
envs = [BulletWrapper(env_, args.__dict__) for env_ in envs]
envEval = BulletWrapper(envEval, args.__dict__)

action_dim = envEval.action_space.shape[0]

"Create config"
if args.keep_seed:
    config = loaded_config
    config['maxStep'] = args.maxStep
else:
    config = OrderedDict(sorted(args.__dict__.items()))
    config['hashCode'] = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['date'] = datetime.now().strftime("%y-%m-%d_%Hh%M_%S")
    config['action_dim'] = action_dim
    config['device'] = device
    config['with_noise'] = args.noise_type != 'none'

with_noise = config['with_noise']
hashCode = config['hashCode']
env_params_name = config['new_env_name'] + ' ' + giveEnv_name(config)
if with_noise:
    noise_adder = AddNoise(config)
else:
    noise_adder = None


if all_dir:
    config['all_dir'] = all_dir
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

if args.dir:
    print('Load: encoder and decoder')
    encoder = torch.load(os.path.join(args.dir, 'state_model.pt'),map_location=torch.device(device))
    if args.method == 'VAE':
        decoder, logvar_fc = torch.load(os.path.join(args.dir, 'state_model_tail.pt'),map_location=torch.device(device))
    else:
        decoder = torch.load(os.path.join(args.dir, 'state_model_tail.pt'),map_location=torch.device(device))
else:
    print('Create: encoder and decoder')
    encoder = CNN(args.state_dim, nc * config['n_stack'], activation=args.activation,
                  debug=args.debug)
    decoder = CNN_Transpose(args.state_dim, nc * config['n_stack'],activation=args.activation, debug=args.debug)
    if args.method == 'VAE':
        lastCNNdim = encoder[0].cnn_layers[-1][-1]
        intermediate_size = encoder[1].out_dim  # the intermediate_size
        layers = [Flatten(lastCNNdim, intermediate_size)]
        layers += [MLP_Module(intermediate_size, layers_MLP + [args.state_dim],activation=args.activation)]
        logvar_fc = torch.nn.Sequential(*layers)

encoder.to(device).train(), decoder.to(device).train()
if args.method == 'VAE': logvar_fc.to(device).train()


Loss_obs = lambda x, y, den, reduction='none': torch.nn.MSELoss(reduction=reduction)(x, y) / (den * config['n_stack'])

if args.method in ['RAE', 'VAE']:
    pp_factor = np.prod(image_size)/args.state_dim
parameters = []
if args.method == 'VAE':
    parameters = list(logvar_fc.parameters())

"create state estimator optimizer"
parameters += list(encoder.parameters())
parametersDecoder = list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=args.lr)
decoderOptimizer = torch.optim.Adam(parametersDecoder, lr=args.lr, weight_decay = args.decoder_weight_lambda)
"force the Garbage Collector to release unreferenced memory"
del parameters, parametersDecoder
gc.collect()

early_stopper = EarlyStopping(patience=args.patience, name=args.method + '-{}'.format(hashCode),
                                   min_delta=0.0001, min_Nepochs=2 * args.patience)
config['early_stop'] = False

"IMPORTANT TO USE FOR CUDA MEMORY"
set_seeds(args.seed)


SRL_name = giveSRL_name(config)
print('SRL_name: %s' % SRL_name)
if all_dir:
    print('all_dir: {}'.format(config['all_dir']))


k_srl = loaded_time_s = 0

start_time = time()
lossAE_log = np.zeros((args.backprop_per_eval), np.float32)
if args.keep_seed:
    elapsed_steps = loaded_config['elapsed_steps']
    elapsed_gradients = loaded_config['elapsed_gradients']
    loaded_time_s = loaded_config['elapsed_time_s']
    elapsed_epochs = loaded_config['elapsed_epochs']
else:
    elapsed_gradients = elapsed_steps = elapsed_epochs = 0

"reset reset info"
dones = [False] * args.num_envs
envs_to_reset = np.array(dones, dtype=np.bool)
reseted = np.arange(args.num_envs)[envs_to_reset]


# Policy configs
with_policy = args.randomExplor and args.env_name in ['HalfCheetahBulletEnv-v0',
                                                                 'InvertedPendulumSwingupBulletEnv-v0']
save_video = True if args.debug else False  # for debugging
if with_policy:
    print('load policy for: {}'.format(args.env_name))
    PiTrained_path = 'PiTrained/PiTrained_{}'.format(config['new_env_name'])
    PiTrained_path = os.path.join(SRL4RL_path, PiTrained_path)
    o_mean, o_std, g_mean, g_std, actor = loadPi(PiTrained_path, model_type='model_best')

"init state with obs without noise"
"reset obs"
if config['n_stack'] > 1:
    obs = np.vstack([env_.reset() for i_, env_ in enumerate(envs)])
    observations = reset_stack(obs, config)
    targets = reset_stack(obs, config)

    obs = envEval.reset()
    observation = reset_stack(obs, config)
    target = reset_stack(obs, config)


if save_video:
    import cv2
    n_video = 0
    video_out = cv2.VideoWriter(
        os.path.join(save_path, '{}_E{:03d}.mp4'.format(args.new_env_name, n_video)),
        cv2.VideoWriter_fourcc(*'mp4v'), fps=5,
        frameSize=(args.image_size, args.image_size))

if args.dir:
    del loaded_config
    gc.collect()

torch.cuda.empty_cache()
"Start AE training"
while not early_stopper.early_stop and elapsed_epochs < args.n_epochs:
    "Do one step with every robots"
    elapsed_steps += 1

    "Make a step"
    has_bump = np.ones((args.num_envs), dtype=np.bool)
    ActEnv = np.zeros((args.num_envs, action_dim), dtype=np.float32)
    while (True in has_bump):
        if with_policy and np.random.uniform(0, 1) < 0.5:
            states = [env_.robot.calc_state() for i_, env_ in enumerate(envs)]
            x = [process_inputs_gt(state_, None, o_mean, o_std, g_mean, g_std) for state_ in states]
            ActEnv[has_bump] = [actor.select_actions(x_, eval=False, goal=None) for x_ in x]
        else:
            ActEnv[has_bump] = np.random.uniform(low=-1, high=1, size=(sum(has_bump), action_dim))
        if args.bumpDetection:
            has_bump = np.array(
                [env_.bump_detection(a_) for env_, a_ in zip(envs[:args.num_envs], ActEnv)])
        else:
            has_bump = np.zeros((args.num_envs), dtype=np.bool)


    TensActions = numpy2pytorch(ActEnv, differentiable=False, device=device)

    "Make a step"
    for step_rep in range(actionRepeat):
        obs, _, dones, _ = zip(*[env_.step(ActEnv[i_]) for i_, env_ in enumerate(envs)])
        "update obs"
        if config['n_stack'] > 1 and args.actionRepeat == 1:
            "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
            observations[:, -2 * nc:] = observations[:, :2 * nc]
            observations[:, :nc] = add_noise(np.vstack(obs), noise_adder, config)
            targets[:, -2 * nc:] = targets[:, :2 * nc]
            targets[:, :nc] = np.vstack(obs)
        elif config['n_stack'] > 1:
            if (step_rep + 1) > (actionRepeat - config['n_stack']):
                observations[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = add_noise(np.vstack(obs),
                                                                                           noise_adder, config)
                targets[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = np.vstack(obs)
        elif (step_rep + 1) == actionRepeat:
            assert step_rep < 2, 'actionRepeat is already performed in env'
            observations = add_noise(np.vstack(obs), noise_adder, config)
            targets = np.vstack(obs)

    "update reset info"
    envs_to_reset = np.array(dones, dtype=np.bool)
    reseted = np.arange(args.num_envs)[envs_to_reset]

    "Compute AE outputs"
    mu = encoder(np2torch(observations))
    if args.beta == 0:
        xHat = decoder_last_layer(decoder(mu))
    else:
        logvar = logvar_fc(encoder[0](np2torch(observations)))
        z = reparametrize(mu, logvar)
        xHat = decoder_last_layer(decoder(z))

    loss = Loss_obs(np2torch(targets), xHat, args.num_envs).sum((1, 2, 3))

    if args.method == 'RAE':
        loss += pp_factor * args.decoder_latent_lambda * ((0.5 * mu.pow(2).sum(-1))) / args.num_envs
    elif args.method == 'VAE':
        loss += pp_factor * args.beta  * (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)) / args.num_envs

    loss = loss.sum()

    assert not torch.isnan(loss).any().item(), 'isnan in loss!!'

    "Update AE network"
    optimizer.zero_grad()
    decoderOptimizer.zero_grad()
    loss.backward()
    decoderOptimizer.step()
    optimizer.step()
    lossAE_log[k_srl] = pytorch2numpy(loss)
    elapsed_gradients += 1
    k_srl += 1

    if save_video:
        update_video(envs[0], im=None, color=args.color, video_size=args.image_size, video=video_out, fpv=args.fpv, downscaling=True)

    if True in dones:
        "update reset observations/states"
        assert args.maxStep != np.inf, 'maxStep == np.inf !'
        "reset obs"
        obs = np.vstack([envs[i_].reset() for i_ in np.arange(args.num_envs)[envs_to_reset]])
        if config['n_stack'] > 1:
            obs = reset_stack(obs, config)
        observations[reseted] = obs
        targets[reseted] = obs.copy()

        if envs_to_reset[0] and save_video:
            n_video += 1
            video_out.release()
            video_out = cv2.VideoWriter(
                os.path.join(save_path, '{}_E{:03d}.mp4'.format(args.new_env_name, n_video)),
                cv2.VideoWriter_fourcc(*'mp4v'), fps=5,
                frameSize=(args.image_size, args.image_size))

    if elapsed_gradients > 1 and (k_srl == args.backprop_per_eval):

        "Plot losses"
        config['elapsed_steps'] = elapsed_steps
        config['elapsed_gradients'] = elapsed_gradients
        config['elapsed_epochs'] = elapsed_epochs = elapsed_gradients // args.backprop_per_eval
        config['elapsed_time_s'] = time() - start_time + loaded_time_s
        config['elapsed_time'] = elapsed_time = str(timedelta(seconds=(int(config['elapsed_time_s']))))

        # Plot loss
        update_text(lossAE_log, save_path + "lossAE_log.txt")
        lossAE_mean = np.mean(lossAE_log)
        config['lossAE_mean'] = float2string(lossAE_mean)
        update_text([lossAE_mean], save_path + "lossAE_mean.txt")
        plotter(lossAE_mean, save_path, name='lossAE_mean', title="{} \n{}".format(SRL_name, hashCode),
                ylabel='Average next obs prediction errors',
                backprop_per_eval=args.backprop_per_eval, text_file=save_path + "lossAE_mean.txt")

        "Real validation for early_stopper"
        with open(save_path + "lossAE_mean.txt") as f:
            loss_log = np.array(f.read().split("\n"), dtype=np.float32)
            lossAE_avg = np.mean(loss_log[-args.patience:])

        print(
            "{} \n[{}]  ~~~~~~ Training only {} {} ~~~~~~\n[{}] Epoch: {} | Gradient steps: {} | {}: {:.3f} | {}: {:.3f}".format(
                env_params_name, hashCode, args.method, SRL_name, elapsed_time, elapsed_epochs, elapsed_gradients,
                'lossAE_mean', lossAE_mean, 'lossAE_avg', lossAE_avg))
        if all_dir:
            print('  all_dir: {}'.format(config['all_dir']))

        early_stopper(-lossAE_avg, Nepochs=elapsed_epochs)
        suffix = 'best' if early_stopper.counter == 0 else 'last'

        "Save models"
        print('  Saving models ......')
        encoder.eval(), decoder.eval()
        save_model(encoder, save_path + 'state_model')
        if args.method == 'VAE':
            logvar_fc.eval()
            save_model([decoder, logvar_fc], save_path + 'state_model_tail')
        else:
            save_model(decoder, save_path + 'state_model_tail')

        if not args.debug:
            nextObsEval = AE_nextObsEval(encoder, decoder,config, save_path, gradientStep=elapsed_gradients,
                                           saved_step=(elapsed_epochs - 1), suffix=suffix,debug=args.debug)
            config['nextObsEval'] = float2string(nextObsEval)
            update_text([nextObsEval], save_path + "nextObsEval_log.txt")
            plotter(nextObsEval, save_path, name='nextObsEval', title="{} \n{}".format(SRL_name, hashCode),
                    ylabel=r'Average obs prediction errors at test-time (after %s updates)' % args.backprop_per_eval,
                    backprop_per_eval=args.backprop_per_eval, text_file=save_path + "nextObsEval_log.txt")

        "Save config.pkl and config.json"
        saveConfig(config, save_dir=save_path)
        saveJson(config, save_path)

        "re-init logs"
        k_srl = 0
        lossAE_log = np.zeros((args.backprop_per_eval), np.float32)

        "Evaluation loop"
        print('  EvalAgent is predicting next obs ......')
        images_hat, images, im_target, im_high_render = np.zeros([evalSteps, *image_size]), np.zeros(
            [evalSteps, *image_size]), np.zeros([evalSteps, *image_size]), np.zeros([evalSteps, 256, 256, 3])
        for evalStep in range(evalSteps):
            with torch.no_grad():
                "Do one step with every robots"
                "Make a step"
                if with_policy and np.random.uniform(0, 1) < 0.5:
                    state = envEval.robot.calc_state()
                    x = process_inputs_gt(state, None, o_mean, o_std, g_mean, g_std)
                    ActEnv = actor.select_actions(x, eval=False, goal=None)
                else:
                    ActEnv = envEval.action_space.sample()
                    TensA = numpy2pytorch(ActEnv, differentiable=False, device=device).unsqueeze(dim=0)
                "Make an evaluation step"
                for step_rep in range(actionRepeat):
                    obs, _, done, _ = envEval.step(ActEnv)
                    "update obs"
                    if config['n_stack'] > 1 and args.actionRepeat == 1:
                        "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
                        observation[:, -2 * nc:] = observation[:, :2 * nc]
                        observation[:, :nc] = add_noise(obs, noise_adder, config)
                        target[:, -2 * nc:] = target[:, :2 * nc]
                        target[:, :nc] = obs
                    elif config['n_stack'] > 1:
                        if (step_rep + 1) > (actionRepeat - config['n_stack']):
                            observation[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = add_noise(obs,
                                                                                    noise_adder,config)
                            target[:, (step_rep - 1) * nc: ((step_rep - 1) + 1) * nc] = obs
                    elif (step_rep + 1) == actionRepeat:
                        assert step_rep < 2, 'actionRepeat is already performed in env'
                        observation = add_noise(obs, noise_adder, config)
                        target = obs

                if done:
                    "reset obs"
                    observation = envEval.reset()
                    if config['n_stack'] > 1:
                        observation = reset_stack(observation, config)
                        target = observation.copy()
                    "save reset"
                    images_hat[evalStep] = np.zeros((image_size), np.float32)
                    images[evalStep] = np.zeros((image_size), np.float32)
                    im_target[evalStep] = np.zeros((image_size), np.float32)
                    im_high_render[evalStep] = np.zeros((256, 256, 3), np.float32)
                    continue

                "Compute AE outputs"
                mu = encoder(np2torch(observation))
                "Reconstruct observations of current step for all trajectories"
                xHat = pytorch2numpy(decoder_last_layer(decoder(mu)))

                "save outputs"
                images_hat[evalStep] = xHat[:, -3:, :, :]
                images[evalStep] = observation[:, -3:, :, :]
                im_target[evalStep] = target[:, -3:, :, :]
                im_high_render[evalStep] = render_env(envEval, 256, False, camera_id_eval, True,
                                                      downscaling=False) / 255.


        "Reconstruct observations for visualization"
        frame = 0
        frames = slice(frame, frame + evalSteps)
        "reshape to HWC"
        images_hat = images_hat[frames].transpose(0, 2, 3, 1)
        images = images[frames].transpose(0, 2, 3, 1)
        im_target = im_target[frames].transpose(0, 2, 3, 1)

        "save last prediction"
        last_idx = -1 if not done else -2
        plot_xHat(images[last_idx], images_hat[last_idx],imgTarget=im_target[last_idx],
                  im_high_render=im_high_render[last_idx], imLabel=imLabel,
                  figure_path=save_path, with_noise=with_noise, with_nextObs=False,
                  gradientStep=elapsed_gradients, suffix=suffix)
        "Reconstruct observations for visualization"
        for i in range(len(images)):
            plot_xHat(images[i], images_hat[i], imgTarget=im_target[i],
                      im_high_render=im_high_render[i], imLabel=imLabel,
                      figure_path=save_path + 'xHat', with_noise=with_noise, with_nextObs=False,
                      gradientStep=elapsed_gradients,
                      saved_step=(i + evalSteps * (elapsed_epochs - 1)))

        print('  end AE evaluation')

        encoder.train(), decoder.train()
        if args.method == 'VAE':
            logvar_fc.train()

        "force the Garbage Collector to release unreferenced memory"
        del images, images_hat, im_high_render
        gc.collect()

if early_stopper.early_stop:
    config['early_stop'] = True
    saveConfig(config, save_dir=save_path)
    saveJson(config, save_path)
