
import torch
from datetime import datetime
import argparse

from SRL4RL.utils.utils import loadConfig, str2bool
from SRL4RL.utils.nn_torch import set_seeds
from SRL4RL.ae.utils import AE_nextObsEval


parser = argparse.ArgumentParser(description='State representation learning new method')
parser.add_argument('--dir', type=str, default='', help='Model path')
parser.add_argument('--debug', type=int, default=0, help='1 for debugging')
parser.add_argument('--renders', type=str2bool, default=False, help='')
args = parser.parse_args()

debug = args.debug
if not debug:
    import matplotlib
    matplotlib.use('Agg')

dir = args.dir
if dir[-1] != '/': dir += '/'
config=loadConfig(dir,name='exp_config')

device = torch.device('cpu')
config['device'] = 'cpu'

if 'n_stack' not in config: config['n_stack'] = 1

args.__dict__.update(config)

encoder = torch.load(dir + 'state_model.pt', map_location=torch.device(device))
if args.method == 'VAE':
    decoder, _ = torch.load(dir+'state_model_tail.pt', map_location=torch.device(device))
else:
    decoder = torch.load(dir+'state_model_tail.pt', map_location=torch.device(device))

encoder.eval(), decoder.eval()

seed_testDataset = datetime.now().microsecond
print('seed for evaluation is: {}'.format(seed_testDataset))
"IMPORTANT TO USE FOR CUDA MEMORY"
set_seeds(seed_testDataset)

nextObsEval = AE_nextObsEval(encoder, decoder,config, dir, suffix='eval',debug=debug)
