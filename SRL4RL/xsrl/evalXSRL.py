import argparse
from datetime import datetime

import gym
import matplotlib
import torch
from bullet_envs.utils import PY_MUJOCO, AddNoise

from SRL4RL.utils.env_wrappers import BulletWrapper
from SRL4RL.utils.nn_torch import set_seeds
from SRL4RL.utils.utils import loadConfig, str2bool
from SRL4RL.xsrl.arguments import is_with_discoveryPi
from SRL4RL.xsrl.utils import XSRL_nextObsEval, getPiExplore, piExplore2obs

# matplotlib.use('nbAgg') # 'nbAgg', 'TkAgg', 'WebAgg'


"register bullet_envs in gym"
import bullet_envs.__init__

parser = argparse.ArgumentParser(description="State representation learning new method")
parser.add_argument("--my_dir", type=str, default="", help="Model path")
parser.add_argument("--debug", type=int, default=0, help="1 for debugging")
parser.add_argument("--renders", type=str2bool, default=False, help="")
parser.add_argument("--nextObsEval_test", type=str2bool, default=True, help="")
args = parser.parse_args()

debug = args.debug
if not debug:
    import matplotlib

    matplotlib.use("Agg")

my_dir = args.my_dir
if my_dir[-1] != "/":
    my_dir += "/"
config = loadConfig(my_dir, name="exp_config")

device = torch.device("cpu")
config["device"] = "cpu"

if "n_stack" not in config:
    config["n_stack"] = 1

args.__dict__.update(config)

alpha, beta, gamma = torch.load(
    my_dir + "state_model.pt", map_location=torch.device(device)
)
omega = torch.load(my_dir + "state_model_tail.pt", map_location=torch.device(device))
alpha.eval(), beta.eval(), gamma.eval(), omega.eval()


seed_testDataset = datetime.now().microsecond
print("seed for evaluation is: {}".format(seed_testDataset))
"IMPORTANT TO USE FOR CUDA MEMORY"
set_seeds(seed_testDataset)

# TODO: remove comments !
if args.nextObsEval_test:
    nextObsEval = XSRL_nextObsEval(
        alpha, beta, gamma, omega, config, my_dir, suffix="evaluate", debug=debug
    )


"evaluate exploration"

maxSteps_visuExplor_env = 1001
if config["env_name"] in ["TurtlebotEnv-v0", "TurtlebotMazeEnv-v0"]:

    assert args.actionRepeat == 1
    visuExplor_env = gym.make(
        config["env_name"],
        renders=False,
        maxSteps=maxSteps_visuExplor_env,
        actionRepeat=config["actionRepeat"],
        image_size=config["image_size"],
        color=True,
        seed=seed_testDataset,
        with_target=False,
        noise_type="none",
        fpv=config["fpv"],
        display_target=False,
        randomExplor=False,
        debug=True,
        wallDistractor=config["wallDistractor"],
        distractor=args.distractor,
    )


elif config["env_name"] in PY_MUJOCO:

    "actionRepeat is done in SRL loop (usually)"
    "actionRepeat is done only one time"
    if config["n_stack"] > 1:
        actionRepeat = args.actionRepeat
        actionRepeat_env = 1
    else:
        actionRepeat = 1
        actionRepeat_env = args.actionRepeat
    visuExplor_env = gym.make(
        "PybulletEnv-v0",
        env_name=config["env_name"],
        renders=False,
        maxSteps=maxSteps_visuExplor_env * config["actionRepeat"],
        actionRepeat=actionRepeat_env,
        image_size=config["image_size"],
        color=True,
        seed=seed_testDataset,
        noise_type="none",
        fpv=config["fpv"],
        doneAlive=False,
        randomExplor=False,
        distractor=args.distractor,
    )


visuExplor_env.seed(seed_testDataset)
"wrap envs"
visuExplor_env = BulletWrapper(visuExplor_env, args.__dict__)

suffix = "_{}".format(seed_testDataset)

if config["with_noise"]:
    noise_adder = AddNoise(config)
else:
    noise_adder = None

with_discoveryPi = is_with_discoveryPi(config)

if with_discoveryPi:
    pi_head, mu_tail, log_sig_tail = torch.load(
        my_dir + "piExplore.pt", map_location=torch.device(device)
    )
    omega = torch.load(
        my_dir + "state_model_tail.pt", map_location=torch.device(device)
    )
    pi_head.eval(), mu_tail.eval(), log_sig_tail.eval()
else:
    pi_head, mu_tail, log_sig_tail = None, None, None

# TODO: remove comments !
piExplore2obs(
    visuExplor_env,
    noise_adder,
    alpha,
    beta,
    gamma,
    omega,
    pi_head,
    mu_tail,
    log_sig_tail,
    config,
    my_dir,
    suffix="evaluate",
    debug=debug,
    evaluate=True,
    saved_step=config["elapsed_epochs"],
)

# TODO: remove comments !
if config["env_name"] in ["TurtlebotEnv-v0", "TurtlebotMazeEnv-v0"]:
    getPiExplore(
        visuExplor_env,
        noise_adder,
        alpha,
        beta,
        gamma,
        pi_head,
        mu_tail,
        log_sig_tail,
        config,
        my_dir,
        debug=debug,
        evaluate=True,
        suffix=suffix,
    )
