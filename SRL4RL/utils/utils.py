

from collections import OrderedDict
from pprint import pprint
import pickle, json
from datetime import datetime

import os

import numpy as np
from collections import defaultdict


state_baselines = ['ground_truth', 'pure_noise', 'position', 'openLoop']
learning_methods = ['RAE','VAE','AE','XSRL']
no_train_encoder_methods = ["random_nn"]
encoder_methods = no_train_encoder_methods + learning_methods

def str2bool(x): return False if x == 'False' else True

def get_hidden(nb_hidden):
    if '-' in nb_hidden:
        return [int(i) for i in nb_hidden.split('-')]
    else:
        try:
            hidden = int(nb_hidden)
            return [hidden] if hidden > 0 else []
        except:
            return []

    
def giveSRL_name(config):
    method_params = 'dim {:02d}'.format(config['state_dim'])
    if config['method'] in learning_methods:
        method_params += ', maxStep {}, num envs {}'.format(config['maxStep'], config['num_envs'])
        method_params += ' randomExplor' if config['randomExplor'] else ''
    return method_params

def give_name(config):
    model_name = config['method']
    factor = 1
    if 'XSRL' in model_name:
        model_name = 'XSRL'
        if 'wEnt' in config['SRL_name']:
            model_name += '-MaxEnt'
        elif 'wLPB' not in config['SRL_name']:
            model_name += '-random'
    elif model_name == 'ground_truth':
        model_name = 'ground truth'
    elif model_name == 'position':
        pass
    elif model_name == 'pure_noise':
        model_name = 'pure noise'
    elif model_name == 'openLoop':
        model_name = 'open-loop'
    elif model_name == 'random_nn':
        model_name = 'random network'
    elif model_name == 'RAE':
        pass
    elif model_name == 'VAE':
        pass
    elif model_name == 'AE':
        pass

    if 'agent' in config: # 'RL_name'
        model_name = 'SAC+'+ model_name
    if 'env_params' in config:
        state_dim = config['env_params']['obs'][0]
    else:
        state_dim = config['state_dim']
    if config['method'] in encoder_methods:
        if config['stack_state']:
            model_name += '-stack'
        if 'randomExplor' in config['SRL_name']:
            model_name += '-explor'
        if config['wallDistractor']:
            model_name+=' (w.distractor)'
    if '(w.distractor)' not in model_name:
        model_name += ' (dim {})'.format(state_dim) if model_name == 'ground truth' else ' (dim %1d)' % (state_dim*factor)
    return model_name


class appendabledict(defaultdict):
    def __init__(self, type_=list, *args, **kwargs):
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)

    def subslice(self, slice_):
        """indexes every value in the dict according to a specified slice
        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.
        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}"""
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        "appends current dict's values with values from other_dict"
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, name="", min_delta=0,baseline=-np.inf,min_Nepochs=0):#save_dir=".models",
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum percentage in the monitored quantity to qualify as an improvement,
                                i.e. an absolute change of less than min_delta, will count as no improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.val_acc_max = 0.
        self.name = name
        self.min_delta = min_delta
        self.baseline = baseline
        self.min_Nepochs = min_Nepochs

    def __call__(self, val_acc, Nepochs=1,model=None,name=None):
        score = val_acc * (1 - np.sign(val_acc) *self.min_delta)
        if score <= self.best_score:
            if Nepochs <= self.min_Nepochs:
                print(f'  EarlyStopping for {self.name} counter: {self.counter} out of {self.patience} | Nepochs: [{Nepochs}/{self.min_Nepochs}]')
            elif val_acc < self.baseline:
                print(f'  EarlyStopping for {self.name} counter: {self.counter} out of {self.patience} | score: {-val_acc}>{-self.baseline}')
            else:
                if self.best_score == np.inf:
                    self.best_score = val_acc
                self.counter += 1
                print(f'  EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'  {self.name} has stopped')

        else:
            self.best_score = val_acc
            self.counter = 0
            return 'best'

    def reset(self,patience):
        self.counter = 0
        self.early_stop = False
        self.best_score = -np.inf
        self.patience = patience


def saveJson(config, save_dir,name='exp_config.json'):
    "Save the experiment config to a json file"
    if name[-5:] != '.json': name += '.json'
    with open(os.path.join(save_dir, name), 'w') as outfile:
        json.dump(config, outfile, sort_keys=True)

def loadJson(curr_path,name='exp_config'):
    if name[-5:] != '.json': name += '.json'
    with open(os.path.join(curr_path,name), "r") as read_file:
        return json.load(read_file)

def saveConfig(config, print_config=False,eval=False,save_dir=None,name='exp_config'):
    "Save the experiment config to a pkl file"
    if name[-4:] != '.pkl': name += '.pkl'
    if not save_dir:
        save_dir = config['log_folder']
    if print_config:
        pprint(config)
        print("Saved config to log folder: {}".format(save_dir))
    # Sort by keys
    config = OrderedDict(sorted(config.items()))
    config_path = "{}/{}".format(save_dir,name)
    if os.path.exists(config_path) and eval:
        config_path = "{}/{}_eval".format(save_dir,name)

    with open(config_path, "wb") as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)


def loadConfig(config_path,name='exp_config'):
    if name[-4:] != '.pkl': name+= '.pkl'
    with open(os.path.join(config_path,name), "rb") as f:
        return pickle.load(f)

def savePickle(dict,path):
    with open(path, 'wb') as f:
        pickle.dump(dict, f)

def loadPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def float2string(value,nfloat=3):
    float2string = '{:.%sf}'%nfloat
    return float2string.format(value)

def update_text(list,save_path,text=False, replace = False, array=False):
    if os.path.isfile(save_path) and not replace:
        assert not array
        mode = "a"
        with open(save_path, mode) as file_object:
            if text:
                file_object.write('\n')
                file_object.write(list)
            else:
                file_object.write('\n')
                file_object.write('\n'.join(map(str, list)))
    else:
        if array:
            if '.txt' == save_path[-4:]: save_path = save_path[:-4]
            np.save(save_path, list)
        else:
            mode = "w"
            with open(save_path, mode) as file_object:
                if text:
                    file_object.write(list)
                else:
                    file_object.write('\n'.join(map(str, list)))

def createFolder(path_to_folder, exist_msg):
    """
    Try to create a folder (and parents if needed)
    print a message in case the folder already exist
    :param path_to_folder: (str)
    :param exist_msg:
    """
    try:
        os.makedirs(path_to_folder)
        print('created folder: {}'.format(path_to_folder))
    except OSError:
        if exist_msg:
            print(exist_msg)
