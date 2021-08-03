import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob2
import argparse
import seaborn as sns

from SRL4RL.utils.utils import loadConfig, give_name, str2bool


matplotlib.use('Agg')

def moving_avg_curve(x,kernel_size=2):
    return np.convolve(x, np.ones(kernel_size) / kernel_size, mode='valid')


def complete_pad(xs, last_value=False,backprop_per_eval=None):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] == maxlen:
            pass
        if last_value:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * np.mean(x[-15:])
        elif backprop_per_eval:
            padding = np.arange(x.shape[0]*backprop_per_eval, maxlen * backprop_per_eval, backprop_per_eval)
        else:
            padding = np.arange(x.shape[0]+1 ,maxlen+1)
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--smooth', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--with_subDir', type=str2bool, default=False, help='')
args = parser.parse_args()


# Load all data.
data = {}
backprop_per_eval_dict = {}

text_file = 'success_rates.txt'

if args.with_subDir:
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', text_file))]
else:
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '*', text_file))]


for curr_path in paths:
    if not os.path.isdir(curr_path):
        continue

    try:
        config = loadConfig(curr_path,name='exp_config.pkl')
        with open(os.path.join(curr_path,text_file)) as f:
            success_rate = np.array(f.read().split("\n"), dtype=np.float32)
    except:
        print('\nskipping {}\n'.format(curr_path))
        continue

    env_id = config['new_env_name']
    model_name = give_name(config)
    if args.verbose:
        print('\ncurr_path: {}, \nhashCode: {}, method: {}, model_name: {}'.format(curr_path, config['hashCode'],
                         config['method'], model_name))

    # Process and smooth data.
    epoch = np.arange(len(success_rate)) + 1

    if 'gradient_steps' not in config: config['gradient_steps'] = config['n_batches']
    if 'eval_interval' not in config: config['eval_interval'] = 1
    backprop_per_eval = config['gradient_steps'] * config['eval_interval']
    gradient_steps = np.arange(0, len(success_rate) * backprop_per_eval, backprop_per_eval)
    steps = gradient_steps
    assert success_rate.shape == steps.shape
    x = steps
    y = success_rate
    if args.smooth:
        # TODO: modify with current experiments
        if env_id == 'TurtlebotMazeEnv':
            kernel_size = 10
            print('moving_avg_curve')
            x, y = moving_avg_curve(x, kernel_size=kernel_size), moving_avg_curve(y, kernel_size=kernel_size)

    # TODO: modify with current experiments
    step_limit = None
    y_min, y_max = 0, 1
    if env_id == 'HalfCheetahBulletEnv':
        y_min, y_max = 7.6, 2646
        step_limit = 2200000
    elif env_id == 'InvertedPendulumSwingupBulletEnv':
        y_min, y_max = -312, 894
        step_limit = 900000
    elif env_id == 'TurtlebotMazeEnv':
        y_min, y_max = 0, 1
        step_limit = 2700000
    y = (y - y_min) / (y_max - y_min)

    if step_limit:
        eval_limit = step_limit / (config['gradient_steps'] * config['eval_interval'])
        eval_limit = int(eval_limit)
        x, y = x[:eval_limit], y[:eval_limit]

    assert x.shape == y.shape

    if env_id not in data:
        data[env_id] = {}
    option = 'RL'
    if option not in data[env_id]:
        data[env_id][option] = {}

    if model_name not in data[env_id][option]:
        data[env_id][option][model_name] = []
    data[env_id][option][model_name].append((x, y))

    backprop_per_eval_dict[env_id] = backprop_per_eval

# TODO: modify with current experiments
list_order = ['ground truth', 'XSRL (d', 'XSRL (w/ distractor)', 'XSRL-MaxEnt', 'XSRL-random','RAE-explor-stack','RAE-explor ','RAE-stack','RAE ',
              'VAE ', 'random network-','random network ','position- ','position ', 'open-loop','pure noise ']

for env_id in data:
    print(data[env_id]['RL'].keys())


for env_id in sorted(data.keys()):
    figure = plt.figure(1, figsize=(7, 3))
    
    print('exporting {}'.format(env_id))
    figure.clf()
    ax = figure.add_subplot(1, 1, 1)
    n_model = 0

    n_method = len(data[env_id][option])
    colors = sns.color_palette("bright", n_colors=n_method)

    for option in sorted(data[env_id].keys()):
        for chosen_model in list_order:
            for nb_model, model_name in enumerate(data[env_id][option].keys()):
                if (chosen_model in model_name):
                    if 'AE' == chosen_model[:2] and 'VAE' in model_name: continue
                    xs, ys = zip(*data[env_id][option][model_name])
                    xs, ys = complete_pad(xs,backprop_per_eval=backprop_per_eval_dict[env_id]), complete_pad(ys,last_value=True)
                    assert xs.shape == ys.shape
                    print(min([min(y_) for y_ in ys]))

                    plot_median = plot_mean =False
                    plot_mean = True
                    if plot_median:
                        plt.plot(xs[0], np.nanmedian(ys, axis=0), color=colors[n_model], label=model_name)
                    elif plot_mean:
                        print(chosen_model)
                        plt.plot(xs[0], np.mean(ys, axis=0), color=colors[n_model], label=model_name)
                    n_model += 1
                    # Shaded area reresents a standard deviation
                    mu=np.mean(ys,axis=0)
                    std = np.std(ys, axis=0)
                    top_std = mu + std
                    below_std = mu - std
                    plt.fill_between(xs[0], below_std, top_std, alpha=0.25)

    start, end = ax.get_xlim()

    # to change the scientific notation in matplotlib
    if end > 1e6:
        sctf_notation = r'($\times$1e6)'
        matplotlib.pyplot.ticklabel_format(axis="both", style="", scilimits=None)
        end_s = '%s' % end
        end = (int(end_s[0]) + 2) * 10 ** (len(end_s) - 1)
        plt.xticks(np.arange(0, end + 1, 200000))
    elif end > 1e5:
        sctf_notation = r'($\times$1e5)'
        matplotlib.pyplot.ticklabel_format(axis="both", style="", scilimits=None)
        end_s = '%s' % end
        end = (int(end_s[0]) + 2) * 10 ** (len(end_s) - 1)
        plt.xticks(np.arange(0, end + 1, 100000)) # 20000 100000
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))  # useMathText=True,
    ax.xaxis.get_offset_text().set_alpha(0) # remove 1e6

    plt.xlabel('training steps {}'.format(sctf_notation),fontsize='large', weight='bold')
    plt.ylabel('mean score',fontsize='large', weight='bold')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')


    plt.grid(True)
    "to get rid of the white space on the abcissa"
    plt.margins(x=0)
    show_goals = False

    if n_method >= 7 and n_method <= 8:
        ax.legend( loc=(-0.1,-0.67), borderaxespad=0.,ncol=2, edgecolor='black', fontsize='large', columnspacing=0.5)
    elif n_method >= 9 and n_method <= 10:
        ax.legend( loc=(-0.1,-0.78), borderaxespad=0.,ncol=2, edgecolor='black',fontsize='large', columnspacing=0.5)
    else:
        ax.legend(loc=(-0.1,-0.56), borderaxespad=0.,ncol=2, edgecolor='black',
                  fontsize='large', columnspacing=0.5)

    if env_id == 'InvertedPendulumSwingupBulletEnv':
        plt.ylim([-0.25, 1])
        plt.yticks([ -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # figure.tight_layout()
    save_path = os.path.join(args.dir, 'plot_success_{}.pdf'.format(env_id))
    print('\nSave figure to : %s\n'%save_path)
    plt.savefig(save_path, bbox_inches='tight', format='pdf',pad_inches=0)

    # Get current size
    fig_size = plt.rcParams["figure.figsize"]

    print("Current size:", fig_size)
    plt.close()

