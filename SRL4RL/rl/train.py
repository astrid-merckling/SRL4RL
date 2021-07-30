import gc
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime
from pprint import pprint

import matplotlib
import torch
from mpi4py import MPI

from SRL4RL import SRL4RL_path
from SRL4RL.rl.arguments import get_args, update_args_RL, assert_args_RL
from SRL4RL.rl.arguments import giveRL_name
from SRL4RL.rl.modules.sac_agent import sac_agent
from SRL4RL.rl.utils.env_utils import make_env, get_env_params, load_config
from SRL4RL.utils.nn_torch import set_seeds
from SRL4RL.utils.utils import createFolder, encoder_methods,learning_methods, giveSRL_name, loadConfig, saveConfig, saveJson
from SRL4RL.utils.utilsEnv import giveEnv_name
from SRL4RL.xsrl.arguments import giveXSRL_name

def launch(config):
    "IMPORTANT TO USE FOR CUDA MEMORY"
    set_seeds(config['seed'] + MPI.COMM_WORLD.Get_rank())
    env, config, runner = make_env(config)
    env_params = get_env_params(env,config)
    config['env_params_name'] = config['new_env_name'] + ' ' + giveEnv_name(config)
    config['env_params'] = env_params
    # create the agent to interact with the environment
    RL_trainer = sac_agent(config,env, env_params,runner)
    torch.cuda.empty_cache()
    "start training from here"
    RL_trainer.learn()
    env.close()


if __name__ == '__main__':
    matplotlib.use('Agg')
    # get the args
    args = get_args()
    "first change args with respect to the env"
    args = update_args_RL(args)

    if args.seed == 123456:
        args.seed = datetime.now().microsecond

    args.random_buffer = True
    if args.dir:
        if args.dir[-1] != '/': args.dir += '/'
        config = loadConfig(args.dir)
        keep_keys = ['dir', 'debug']

        select_new_args = {k: args.__dict__[k] for k in keep_keys}
        config.update(select_new_args)
        config['random_buffer'] = False
        if config['srl_path']:
            "update srl_path"
            config['srl_path'] = args.dir

        "force the Garbage Collector to release unreferenced memory"
        del select_new_args, keep_keys
        gc.collect()

    elif args.srl_path:
        args.with_images = True
        "Cannot learn from images without SRL model!"
        args = load_config(args)

    args = assert_args_RL(args)


    if args.dir:
        config['save_dir'] = args.dir
    else:
        # Building the experiment config file
        config = OrderedDict(sorted(args.__dict__.items()))
        hashCode = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        # Path where all the output files are saved
        save_path = os.path.join(SRL4RL_path, config['logs_dir'])
        # Update config with information related to the experiment
        config['RL_name'] = giveRL_name(config)
        config['date'] = datetime.now().strftime("%y-%m-%d_%Hh%M_%S")
        config['hashCode'] = hashCode

        "Create folder, save config"
        if save_path[-1] != '/': save_path += '/'
        save_path += '%s' % config['hashCode'] + '/'
        config['save_dir'] = save_path
        createFolder(save_path, "save_path folder already exist")

    if config['method'] in encoder_methods:
        if config['dir'] and config['srl_path']:
            srl_config = loadConfig(config['srl_path'], name='srl_config')
            config['SRL_name'] = giveXSRL_name(srl_config) if 'XSRL' in srl_config['method'] else giveSRL_name(
                srl_config)
            del srl_config
        elif config['srl_path'] and (config['method'] in learning_methods):
            srl_config = loadConfig(config['srl_path'])
            saveConfig(srl_config, save_dir=save_path, name='srl_config')
            saveJson(srl_config, save_path, 'srl_config')
            config['SRL_name'] = giveXSRL_name(srl_config) if 'XSRL' in srl_config['method'] else giveSRL_name(
                srl_config)
            del srl_config
        else:
            config['SRL_name'] = giveSRL_name(config)

    del args
    gc.collect()
    config = OrderedDict(sorted(config.items()))
    pprint(config)


    launch(config)

