import argparse

from bullet_envs.utils import PY_MUJOCO, env_with_goals, NOISES, PYBULLET_ENV, ENV_ALIVE

from SRL4RL.utils.utils import state_baselines, encoder_methods, str2bool
from SRL4RL.utils.utilsEnv import update_args_envs, assert_args_envs


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", type=str2bool, default=True, help="If True, do not use resize"
    )
    parser.add_argument("--debug", type=int, default=0, help="1 for debugging")
    "Logs hyper-parameters"
    parser.add_argument("--logs_dir", type=str, default="logsRL", help="")
    parser.add_argument(
        "--method",
        type=str,
        default="ground_truth",
        choices=encoder_methods + state_baselines,
        help="Method to use for training representations",
    )
    parser.add_argument("--seed", type=int, default=123456, help="random seed")
    "Loading pretrained models"
    parser.add_argument(
        "--dir", type=str, default="", help="path of the model to continue training"
    )
    parser.add_argument(
        "--srl_path",
        type=str,
        default="",
        help="The SRL pretrained model to tranform image observations to states",
    )
    "Environment hyper-parameters"
    parser.add_argument(
        "--env_name",
        type=str,
        default="TurtlebotMazeEnv-v0",
        choices=PYBULLET_ENV,
        help="the environment name",
    )
    parser.add_argument(
        "--render", type=str2bool, default=False, help="Create viewer for rendering"
    )
    parser.add_argument(
        "--bufferCapacity",
        type=int,
        default=int(1e5),
        help="the number of stored steps in the buffer (for doneAlive)",
    )
    parser.add_argument(
        "--random_target",
        type=str2bool,
        default=True,
        help="Change position of goal at every reset (only for envs in env_with_goals)",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=0,
        help="Number of time steps per episode",
    )
    parser.add_argument(
        "--actionRepeat",
        type=int,
        default=1,
        help="Number of frame skip, i.e. action repeat",
    )
    parser.add_argument(
        "--noise_type", type=str, default="none", choices=NOISES, help="Add some noise"
    )
    parser.add_argument(
        "--flickering",
        type=float,
        default=0.0,
        help=">0 for flickering in env rendering, only for XSRL training",
    )
    parser.add_argument(
        "--distractor",
        type=str2bool,
        default=False,
        help="Add distractor in environment, only for ReacherBulletEnv",
    )
    parser.add_argument(
        "--wallDistractor",
        type=str2bool,
        default=False,
        help="Whether or not include a stochastic visual distractor, only for TurtlebotMazeEnv",
    )
    parser.add_argument(
        "--fpv",
        type=str2bool,
        default=False,
        help="Use FPV camera of the robot (only for envs in env_with_fpv)",
    )
    "SAC options"
    parser.add_argument(
        "--agent",
        type=str,
        default="SAC",
        choices=["SAC"],
        help="Only SAC RL method is available",
    )
    parser.add_argument(
        "--n_eval_rollouts",
        type=int,
        default=10,
        help="the number of episodes for evaluation",
    )
    parser.add_argument(
        "--polyak", type=float, default=0.02, help="polyak averaging coefficient"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parser.add_argument(
        "--init_temperature",
        type=float,
        default=0.1,
        help="entropy coefficient (Temperature parameter α determines the relative importance of the entropy term against the reward)",
    )
    parser.add_argument(
        "--automatic_entropy_tuning", type=str2bool, default=True, help="Tune entropy"
    )
    "NN hyper-parameters : (actor and critic)"
    parser.add_argument(
        "--linearApprox",
        type=str2bool,
        default=False,
        help="approximate with linear networks",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=2,
        help="Number of layers in mu_tail and log_sig_tail",
    )
    parser.add_argument(
        "--nb_hidden",
        type=str,
        default="128-512-128",
        help="number of hidden units per layer",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=20,
        help="Size of features only for `random_nn` method",
    )
    parser.add_argument(
        "--activation",
        default="leaky_relu",
        choices=["leaky_relu", "relu", "elu", "tanh", "tanh_sigmoid"],
        help="Activation to use",
    )
    parser.add_argument(
        "--weight_init",
        type=str,
        default="none",
        choices=["xavier", "orthogonal", "random_init", "random_init_trunc", "none"],
        help="Method to use for weights initialization in Actor and Critic",
    )
    "State representation options"
    parser.add_argument(
        "--stack_state",
        type=str2bool,
        default=False,
        help="if True then stack two consecutive states (only with SRL models)",
    )
    parser.add_argument(
        "--double_state",
        type=str2bool,
        default=False,
        help="Stack consecutive states to double dimension (to test the influence of the input dimension with SAC)",
    )
    "Input preprocessing"
    parser.add_argument(
        "--clip_range",
        type=float,
        default=10,
        help="normalized observations are cropped to this values",
    )
    parser.add_argument("--clip_obs", type=float, default=200, help="the clip ratio")
    "SGD hyper-parameters"
    parser.add_argument(
        "--n_episodes_rollout", type=int, default=2, help="the rollouts per mpi thread"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000000,
        help="the number of epochs to train the agent",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=2,
        help="update the target network every ```gradient_step+1`` % ``target_update_interval``.",
    )
    parser.add_argument(
        "--actor_update_interval",
        type=int,
        default=2,
        help="update the actor network and alpha every ```gradient_step+1`` % ``actor_update_interval``.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="the sample batch size"
    )
    parser.add_argument(
        "--lr_actor", type=float, default=5e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--lr_critic", type=float, default=5e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--lr_alpha", type=float, default=5e-4, help="the learning rate of the entropy"
    )

    args = parser.parse_args()
    return args


def update_args_RL(args):
    args = update_args_envs(args)
    args.doneAlive = True if args.env_name in ENV_ALIVE else False

    if args.max_episode_steps == 0:
        args.max_episode_steps = (
            1000 if args.env_name in PY_MUJOCO else args.max_episode_steps
        )
        args.max_episode_steps = (
            50 if "Reacher" in args.env_name else args.max_episode_steps
        )
        args.max_episode_steps = (
            100 if "TurtlebotMazeEnv" in args.env_name else args.max_episode_steps
        )
        args.max_episode_steps = (
            50 if args.env_name == "TurtlebotEnv-v0" else args.max_episode_steps
        )

    args.n_eval_rollouts = int(10)
    args.minEpBuffer = 32
    "How many gradient steps to do after each rollout `n_episodes_rollout`"
    args.gradient_steps = int(64)

    if args.env_name in [
        "TurtlebotEnv-v0",
        "TurtlebotMazeEnv-v0",
        "ReacherBulletEnv-v0",
        "InvertedPendulumSwingupBulletEnv-v0",
    ]:
        args.patience = 300
        args.eval_interval = int(10)
        if args.env_name in ["TurtlebotEnv-v0"]:  # , 'ReacherBulletEnv-v0'
            args.nb_hidden = "256"
            args.cutoff = 0
    elif args.env_name in PY_MUJOCO:
        args.eval_interval = int(50)

        """kostrikov2020DrQ use same parameters: 
        for HalfCheetahBulletEnv-v0:
            init_temperature=0.1, batch_size=256, actionRepeat=4
        """
        args.patience = (
            200 if args.env_name == "InvertedPendulumSwingupBulletEnv-v0" else 50
        )  # changed during RL_training

    if args.debug:
        args.max_episode_steps = int(64)
        args.n_eval_rollouts = int(4)
        args.nb_layer = 1
        # args.n_epochs = 1000
        args.bufferCapacity = 1000
        args.minEpBuffer = 4

    if args.method in state_baselines:
        args.with_images = False
        args.n_stack = 1
    else:
        args.with_images = True

    args.gamma = args.gamma ** args.actionRepeat
    args.numEpBuffer = args.bufferCapacity // (
        args.max_episode_steps // args.actionRepeat
    )

    return args


def assert_args_RL(args):
    assert_args_envs(args)
    if args.noise_type != "none":
        assert args.with_images, "noise in image observations with ground_truth"
    if args.double_state:
        assert not args.stack_state
    if args.stack_state:
        assert args.with_images

    args.random_target = (
        args.random_target if (args.env_name in env_with_goals) else False
    )
    args.with_goal = args.random_target

    if args.method == "position":
        assert args.env_name in [
            "InvertedPendulumSwingupBulletEnv-v0"
        ], "method=position with env not requiring velocities"

    return args


def giveRL_name(config):
    if "double_state" not in config:
        config["double_state"] = False
    RL_name = config["agent"]
    RL_name += " stack" if config["stack_state"] else ""
    RL_name += " double-state" if config["double_state"] else ""
    return RL_name
