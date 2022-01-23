import argparse

from bullet_envs.utils import PYBULLET_ENV, NOISES

from SRL4RL.utils.utils import str2bool
from SRL4RL.utils.utilsEnv import update_args_envs, assert_args_envs
import numpy as np


"""
Here are the params for training XSRL models
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="XSRL (eXploratory State Representation Learning)"
    )
    parser.add_argument(
        "--evalExplor",
        type=str2bool,
        default=False,
        help="for exploration evaluation during training of 550 max steps",
    )
    parser.add_argument("--debug", type=int, default=0, help="1 for debugging")
    "Logs hyper-parameters"
    parser.add_argument(
        "--logs_dir", type=str, default="logsXSRL", help="path where to save the models"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="XSRL",
        help="Method to use for training representations",
    )
    parser.add_argument(
        "--seed", type=int, default=123456, help="Random seed to use"
    )  # 123456
    parser.add_argument(
        "--keep_seed", type=str2bool, default=False, help="only with dir"
    )
    "Loading pretrained models"
    parser.add_argument(
        "--dir", type=str, default="", help="path of the pretrained model to initialize"
    )
    parser.add_argument(
        "--reset_policy",
        type=str2bool,
        default=False,
        help="only with dir: reset policy",
    )
    "Environment hyper-parameters"
    parser.add_argument(
        "--env_name",
        type=str,
        default="TurtlebotMazeEnv-v0",
        choices=PYBULLET_ENV,
        help="Environment name",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of envs to train in parallel, corresponds to the batch size",
    )
    parser.add_argument(
        "--randomExplor",
        type=str2bool,
        default=False,
        help="Always use a constant reset state",
    )
    parser.add_argument(
        "--maxStep", type=float, default=500, help="maxStep for training env"
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

    "XSRL options"
    parser.add_argument(
        "--resetPi",
        type=str2bool,
        default=False,
        help="resetPi trick for better diversity: train two discovery policies",
    )
    parser.add_argument(
        "--dvt_patience", type=int, default=2, help="Max counts before to resetPi"
    )
    parser.add_argument("--weightInverse", type=float, default=0.0, help="weight")
    parser.add_argument("--weightLPB", type=float, default=0.0, help="weight")
    parser.add_argument("--weightEntropy", type=float, default=0.0, help="weight")
    parser.add_argument(
        "--autoEntropyTuning", type=str2bool, default=False, help="Tune weightEntropy"
    )
    parser.add_argument(
        "--init_temperature",
        type=float,
        default=0.1,
        help="init entropy coefficient for autoEntropyTuning (Temperature parameter α determines the relative importance of the entropy term against the reward)",
    )
    "NN hyper-parameters : (alpha, beta, gamma)"
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
        choices=["xavier", "orthogonal", "random_init", "random_init_trunc"],
        help="Method to use for weights initialization",
    )
    parser.add_argument(
        "--alpha_dim", type=int, default=30, help="alpha output dimension"
    )
    parser.add_argument("--beta_dim", type=int, default=0, help="beta output dimension")
    parser.add_argument("--state_dim", type=int, default=5, help="state dimension")
    parser.add_argument(
        "--nb_hidden_gamma",
        type=str,
        default="128-512-128",
        help="number of hidden units per layer",
    )
    parser.add_argument(
        "--nb_hidden_pi",
        type=str,
        default="128-512-128",
        help="number of hidden units per layer",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=2,
        help="Number of layers in mu_tail and log_sig_tail",
    )
    "SGD hyper-parameters"
    parser.add_argument(
        "--pi_batchSize",
        type=int,
        default=128,
        help="pi_batchSize for updating the discovery policy and inverse/LPB-target models",
    )
    parser.add_argument(
        "--pi_updateInt", type=int, default=512, help="maxStep for training env"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "amsgrad", "adamW"],
        help="for automatic weights regularization",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--lr_explor", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--lr_alpha", type=float, default=1e-4, help="Learning rate for training"
    )
    parser.add_argument(
        "--n_epochs", type=float, default=1e6, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=40,
        help="Number of epochs before early_stopping",
    )
    parser.add_argument(
        "--backprop_per_eval",
        type=int,
        default=2048,
        help="Number of iterations per evaluation",
    )
    args = parser.parse_args()

    return args


def assert_args_XSRL(args):
    assert_args_envs(args)

    if args.reset_policy:
        assert args.dir, "reset_policy without --dir"
    if args.keep_seed:
        assert args.dir

    args.nEnv_perPi = args.num_envs

    with_discoveryPi = is_with_discoveryPi(args.__dict__)
    if with_discoveryPi:
        if args.resetPi:
            args.nEnv_perPi = args.num_envs // 2
        assert (
            args.pi_batchSize % args.nEnv_perPi == 0
        ), "need nPi_samples over each num_envs to have pi_batchSize"
        args.nPi_samples = args.pi_batchSize // args.nEnv_perPi
        args.pi_updateInt = max(args.nPi_samples, args.pi_updateInt)
        assert (
            args.backprop_per_eval % args.pi_updateInt == 0
        ), "there are pi_backprop_per_eval per epoch"
        args.pi_backprop_per_eval = args.backprop_per_eval // args.pi_updateInt
        assert args.pi_updateInt % args.nPi_samples == 0, (
            "there are nPi_samples per pi_updateInt"
            "there are pi_sampling steps between each sample"
        )
        args.pi_sampling = args.pi_updateInt // args.nPi_samples
        print("args.pi_updateInt", args.pi_updateInt)
        print("args.nPi_samples", args.nPi_samples)
        print("args.pi_sampling", args.pi_sampling)
        print("args.pi_backprop_per_eval", args.pi_backprop_per_eval)
    else:
        assert not args.resetPi
        assert not args.reset_policy

    return args


def update_args_XSRL(args):
    args = update_args_envs(args)

    if args.maxStep > 9e4:
        args.maxStep = np.inf
    args.inverse = args.weightInverse > 0
    args.LPB = args.weightLPB > 0
    args.entropy = args.weightEntropy > 0 or args.autoEntropyTuning

    if args.env_name == "TurtlebotEnv-v0":
        args.nb_hidden_pi = "256"
        args.cutoff = 1
    return args


def is_with_discoveryPi(config):
    with_discoveryPi = config["inverse"] or config["LPB"] or config["entropy"]
    return with_discoveryPi


def giveXSRL_name(config):
    with_discoveryPi = is_with_discoveryPi(config)
    method_params = ""
    method_params += "randomExplor" if config["randomExplor"] else ""
    method_params += (
        "[{}] ".format(config["noise_type"]) if config["noise_type"] != "none" else ""
    )
    method_params += (
        "[flickering-{}] ".format(config["flickering"])
        if config["flickering"] > 0
        else ""
    )
    method_params += "dim {:02d}, maxStep {}, num envs {}, ".format(
        config["state_dim"], config["maxStep"], config["num_envs"]
    )
    if with_discoveryPi:
        method_params += "batchPi {}, pi_updateInt {}, ".format(
            config["pi_batchSize"], config["pi_updateInt"]
        )
    method_params += (
        "dvt_patience-{}, ".format(config["dvt_patience"]) if config["resetPi"] else ""
    )
    method_params += (
        "wInverse-{}, ".format(config["weightInverse"])
        if config["weightInverse"] > 0
        else ""
    )
    method_params += (
        "wLPB-{}, ".format(config["weightLPB"]) if config["weightLPB"] > 0 else ""
    )
    method_params += (
        "optim-{}, ".format(config["optimizer"])
        if config["optimizer"] != "adam"
        else ""
    )
    method_params += (
        "hidden-{}, ".format(config["nb_hidden_gamma"])
        if config["nb_hidden_gamma"] != "128-512-128"
        else ""
    )
    if with_discoveryPi:
        if config["autoEntropyTuning"]:
            method_params += "autoEnt, "
        else:
            method_params += (
                "wEnt-{}, ".format(config["weightEntropy"])
                if config["weightEntropy"] > 0
                else ""
            )
    method_params = method_params[:-2]
    return method_params
