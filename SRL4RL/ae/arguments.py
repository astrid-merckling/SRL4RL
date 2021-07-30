

import argparse

from bullet_envs.utils import PYBULLET_ENV, NOISES

from SRL4RL.utils.utils import str2bool
from SRL4RL.utils.utilsEnv import update_args_envs, assert_args_envs
import numpy as np


AE_variants = ['AE', 'RAE' , 'VAE', ]

"""
Here are the params for training AE variants
"""

def get_args():

    parser = argparse.ArgumentParser(description='AE (autoencoder')
    parser.add_argument('--debug', type=int, default=0, help='1 for debugging')
    "Logs hyper-parameters"
    parser.add_argument("--dir", type=str, default="", help='path of the pretrained model to initialize')
    parser.add_argument('--logs_dir', type=str, default='logs', help='path where to save the models')
    parser.add_argument('--method', type=str, default='RAE', choices=AE_variants,
                        help='Method to use for training representations')
    parser.add_argument('--seed', type=int, default=123456, help='Random seed to use')  # 123456
    parser.add_argument('--keep_seed', type=str2bool, default=False, help='only with dir')
    "Environment hyper-parameters"
    parser.add_argument('--env_name', type=str, default='TurtlebotMazeEnv-v0', choices=PYBULLET_ENV,
                        help='Environment name')
    parser.add_argument("--num_envs", type=int, default=32,
                        help="Number of envs to train in parallel, corresponds to the batch size")
    parser.add_argument('--randomExplor', type=str2bool, default=False,
                        help='Use effective exploration with random init on Turtle; with SAC trained policies otherwise')
    parser.add_argument('--maxStep', type=float, default=500, help='maxStep for training env')
    parser.add_argument('--actionRepeat', type=int, default=1, help='Number of frame skip, i.e. action repeat')
    parser.add_argument("--noise_type", type=str, default='none', choices=NOISES, help='Add some noise')
    parser.add_argument('--flickering', type=float, default=0., help='>0 for flickering in env rendering, only for XSRL training')
    parser.add_argument("--distractor", type=str2bool, default=False,
                        help='Add distractor in environment, only for ReacherBulletEnv')
    parser.add_argument('--wallDistractor', type=str2bool, default=False,
                        help='Whether or not include a stochastic visual distractor, only for TurtlebotMazeEnv')
    parser.add_argument('--fpv', type=str2bool, default=False,
                        help='Use FPV camera of the robot (only for envs in env_with_fpv)')
    "NN hyper-parameters"
    parser.add_argument('--activation', default='leaky_relu',
                        choices=['leaky_relu', 'relu', 'elu', 'tanh', 'tanh_sigmoid'],
                        help='Activation to use')
    parser.add_argument('--weight_init', type=str, default='none',
                        choices=['xavier', 'orthogonal', 'random_init', 'random_init_trunc'],
                        help='Method to use for weights initialization')
    parser.add_argument('--state_dim', type=int, default=5, help='state dimension')
    "SGD hyper-parameters"
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'amsgrad', 'adamW'],
                        help='for automatic weights regularization')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument("--n_epochs", type=float, default=1e6, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=40, help="Number of epochs before early_stopping")
    parser.add_argument('--backprop_per_eval', type=int, default=2048, help='Number of iterations per evaluation')
    args = parser.parse_args()


    return args

def assert_args_AE(args):
    assert_args_envs(args)

def update_args_AE(args):
    args = update_args_envs(args)
    if args.logs_dir == 'logs':
        args.logs_dir += args.method

    if args.method == 'RAE':
        args.decoder_weight_lambda = 1e-7
        args.decoder_latent_lambda = 1e-6
    else:
        args.decoder_weight_lambda = 0
        args.decoder_latent_lambda = 0

    if args.maxStep > 9e4:
        args.maxStep = np.inf

    if args.method in ['AE', 'RAE']:
        args.beta = 0
    elif args.method == 'VAE':
        args.beta = 1

    return args