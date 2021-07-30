import gym
from mpi4py import MPI
import torch

from SRL4RL.rl.utils.runner import StateWrapper, Runner, RandomNetworkRunner
from SRL4RL.xsrl.utils import XSRLRunner
from SRL4RL.ae.utils import AERunner
from SRL4RL.utils.env_wrappers import  BulletWrapper, GoalWrapper
from SRL4RL.utils.utils import loadConfig, encoder_methods

from bullet_envs.utils import env_with_goals, PY_MUJOCO
"register bullet_envs in gym"
import bullet_envs.__init__


def load_config(args):
    print('\nsrl_path: ', args.srl_path)
    args.srl_path = args.srl_path[:-1] if args.srl_path[-1] == '/' else args.srl_path
    srl_config = loadConfig(args.srl_path)

    # New RL arguments:
    option = 'RL' if 'n_eval_rollouts' in srl_config else 'SRL'

    SRL_args = ['method', 'n_stack', 'image_size', 'env_name', 'fpv', 'state_dim', 'color',
                'noise_type', 'activation', 'wallDistractor', 'actionRepeat']

    select_SRL_args = {k: srl_config[k] for k in SRL_args}
    args.__dict__.update(select_SRL_args)
    if option == 'SRL':
        args.E_hashCode = srl_config['hashCode']
    else:
        args.E_hashCode = srl_config['hashCode'] + '_RL'

    del srl_config
    return args


def make_env(config,):
    seed = config['seed']
    seed += MPI.COMM_WORLD.Get_rank()

    # parameters for demo.py:
    if 'display_target' in config:
        display_target = config['display_target']
    else:
        display_target = False
    if config['with_images']:
        color = config['color']
    else:
        color = False
    nc = 3 if color else 1

    if 'demo' in config:
        noise_type = config['noise_type']
    else:
        noise_type = 'none'  # noise added in add_noise()
    if 'n_stack' not in config:
        config['n_stack'] = 1
        print('\nn_stack not in args !\n')

    target_pos = None
    if config['env_name'] in env_with_goals and 'target_pos' in config:
        target_pos = config['target_pos']
    if config['env_name'] in PY_MUJOCO:
        # TODO: verify that actionRepeat is not performed 2 times (i.e. by env and by runner)
        actionRepeat = 1 if config['n_stack'] > 1 else config['actionRepeat']
        # actionRepeat = 1
        env = gym.make('PybulletEnv-v0', env_name=config['env_name'], renders=config['render'], distractor=config['distractor'],
                       maxSteps=config['max_episode_steps'], actionRepeat=actionRepeat,
                       image_size=config['image_size'], color=color, seed=seed,
                       noise_type=noise_type, fpv=config['fpv'], doneAlive=config['doneAlive'],
                       randomExplor=True,
                       random_target=config['random_target'], target_pos=target_pos, display_target=display_target)
    elif config['env_name'] in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:
        env = gym.make(config['env_name'], renders=config['render'],
                       distractor=config['distractor'],
                       maxSteps=config['max_episode_steps'], actionRepeat=config['actionRepeat'],
                       image_size=config['image_size'], color=color, seed=seed,
                       noise_type=noise_type, fpv=config['fpv'],
                       randomExplor=True, wallDistractor=config['wallDistractor'],
                       random_target=config['random_target'], target_pos=target_pos, display_target=display_target,)
    # wrap env
    env = BulletWrapper(env, config)

    print('Env seed for rank {} is: {}'.format(MPI.COMM_WORLD.Get_rank(), seed))
    if config:
        if not config['random_target'] and (config['env_name'] in env_with_goals):
            env.reset()
            config['target_pos'] = env.target
        else:
            config['target_pos'] = None

    demo = False
    if 'demo' in config:
        demo = True
        config['device'] = 'cpu'

    if config['method'] in encoder_methods and config['cuda']:
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        print('device for StateRunner is: %s' % config['device'])
    else:
        config['device'] = 'cpu'

    config['action_dim'] = env.action_space.shape[0]

    # create_runner
    if 'XSRL' in config['method']:
        runner = XSRLRunner(config)
    elif config['method'] in ['RAE', 'VAE', 'AE']:
        runner = AERunner(config)
    elif config['method'] == 'random_nn':
        runner = RandomNetworkRunner(nc * config['n_stack'], config)
    else:
        runner = Runner()

    # last env wrappers
    if config['with_images']:
        env = StateWrapper(env, runner, demo)
    if config['with_goal']:
        env = GoalWrapper(env)

    if config['method'] in encoder_methods:
        runner.train(training=False)
        runner.to_device(device=config['device'])

    return env, config, runner


def get_env_params(env, config):
    obs = env.reset()
    if config['method'] in encoder_methods:
        if config['double_state'] or config['stack_state']:
            obs_shape = config['state_dim'] * 2
        else:
            obs_shape = config['state_dim']
        obs_shape = tuple((obs_shape,))
    else:
        obs_shape = obs['observation'].shape[-1] if config['with_goal'] else obs.shape[-1]
        obs_shape = tuple((obs_shape,))

    goal_shape = obs['desired_goal'].shape[0] if config['with_goal'] else 0
    action_dim = env.action_space.shape[0]
    params = {'obs': tuple(obs_shape),
              'goal': goal_shape,
              'action': action_dim,
              'action_max': env.action_space.high[0].item(),
              'action_min': env.action_space.low[0].item(),
              }
    print('env params: {}'.format(params))
    return params
