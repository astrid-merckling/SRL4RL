
import torch
from torch.autograd import Variable
import numpy as np
import random


layers_alpha = [('/', 2, 32), ('/', 2, 64), ('/', 2, 128), ('/', 2, 256), ]
layers_alpha_debug = [('/', 2, 1), ('/', 2, 1), ('/', 2, 1), ('/', 2, 1), ]

layers_cnn_omega = [('*', s, channels) for (c, s, channels) in layers_alpha[::-1]]

layers_MLP=[256, 32]
layers_MLP_omega = layers_MLP[::-1]

def save_model(model,save_dir):
    if save_dir[-3:] != '.pt': save_dir+= '.pt'
    torch.save(model, save_dir)


def set_seeds(seed):
    "IMPORTANT TO USE FOR CUDA MEMORY"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def numpy2pytorch(x, differentiable=True, dtype=torch.float, device=None):
    if differentiable:
        if device is None:
            res = torch.nn.Parameter(Variable(torch.from_numpy(x), True).type(dtype))
        else:
            res = torch.nn.Parameter(Variable(torch.from_numpy(x), True).type(dtype).to(device))
    else:
        if device is None:
            res = Variable(torch.from_numpy(x), False).type(dtype)
        else:
            res = Variable(torch.from_numpy(x), False).type(dtype).to(device)
    return res


def pytorch2numpy(x):
    return x.cpu().detach().numpy()
def pytorch2item(x):
    return x.cpu().detach().item()

def conv2d_size_out(size, kernel_size=5, stride=2,padding=2):
    return (size +2*padding- (kernel_size - 1) - 1) // stride + 1


# Exemple of CNN
def CNN(out_size, n_channels,obs_shape=64,activation='leaky_relu',weight_init='none',requires_grad = True, debug = 0):

    if debug:
        "to save computational memory"
        layers_cnn = [('*', s, 1) for (c, s, channels) in layers_alpha]
        layers_hidden = [1, 1]
        intermediate_size = 1
    else:
        layers_cnn = layers_alpha
        layers_hidden = layers_MLP
        intermediate_size = 1024
    layers = []
    layers += [CNN_Module(n_channels, layers_cnn,obs_shape=obs_shape,activation=activation,with_last_actv=True)]
    layers += [Flatten(layers_cnn[-1][-1], intermediate_size)]
    layers += [MLP_Module(intermediate_size, layers_hidden + [out_size],activation=activation)]
    cnn = torch.nn.Sequential(*layers)
    if weight_init != 'none':
        weight_init_ = import_weight_init(weight_init)
        cnn.apply(weight_init_)

    if not requires_grad:
        for p in cnn.parameters():
            p.requires_grad = False
    return cnn


# Exemple of CNN_Transpose
def CNN_Transpose(feature_size, n_channels, obs_shape=64,probabilistic=False,cutoff=2,activation='leaky_relu',debug=0):
    # in probabilistic settings mean and covariance share all layers up to cutoff
    if debug:
        "to save computational memory"
        layers_cnn = [('*', s, 1) for (c, s, channels) in layers_alpha]
        layers_cnn_transpose = [('*', s, 1) for (c, s, channels) in layers_cnn_omega]
        layers_hidden = [1, 1]
        intermediate_size = 1
    else:
        layers_cnn = layers_alpha
        layers_cnn_transpose = layers_cnn_omega
        layers_hidden = layers_MLP_omega
        intermediate_size = 1024
    
    layers_transpose = layers_cnn_transpose[1:]

    layers_head = []
    layers_head += [MLP_Module(feature_size, layers_hidden + [intermediate_size],activation=activation)]
    layers_head += [Unflatten(intermediate_size, layers_cnn[-1][-1], 4, 4,activation=activation)]

    layers_head += [CNN_Module(layers_cnn[-1][-1], layers_transpose[:-cutoff], obs_shape=obs_shape,activation=activation,with_last_actv=True)]

    head_dim = layers_transpose[len(layers_transpose)-cutoff-1][-1]
    mu_tail = CNN_Module(head_dim, layers_transpose[-cutoff:] + [('*', 2, n_channels)],obs_shape=obs_shape,activation=activation)
    if probabilistic:
        logitSig_tail = CNN_Module(head_dim, layers_transpose[-cutoff:] + [('*', 2, 1)],obs_shape=obs_shape,activation=activation)
        return torch.nn.Sequential(*layers_head), mu_tail, logitSig_tail
    else:
        layers_head += [mu_tail]
        return torch.nn.Sequential(*layers_head)

def MLP_mdn(in_dim,dimensions,cutoff=2,activation='leaky_relu',weight_init='none',requires_grad=True, MDN=False):
    assert len(dimensions)>=cutoff, 'Not enough dimensions for MDN network!'

    mdn_head = MLP_Module(in_dim, dimensions[:-cutoff],activation=activation,
                                   with_last_actv=True, requires_grad=requires_grad)
    head_dim=dimensions[len(dimensions)-cutoff-1]
    mu_tail = MLP_Module(head_dim, dimensions[-cutoff:],activation=activation, requires_grad=requires_grad)
    sig_tail = MLP_Module(head_dim, dimensions[-cutoff:],activation=activation, requires_grad=requires_grad)
    if weight_init != 'none':
        weight_init_ = import_weight_init(weight_init)
        sig_tail.apply(weight_init_)
    if MDN:
        logitPi_tail = MLP_Module(head_dim, dimensions[-cutoff:],activation=activation,requires_grad=requires_grad)
    else:
        logitPi_tail = None
    return mdn_head, mu_tail, sig_tail, logitPi_tail


# Module to flatten images into vectors
class Flatten(torch.nn.Module):
    def __init__(self, in_channels, out_dim, ):
        super(Flatten, self).__init__()
        self.out_dim = out_dim
        conv = torch.nn.Conv2d(in_channels, out_dim, kernel_size=(5, 5), padding=2, bias=True)
        self.conv = conv

    def forward(self, x):
        out = self.conv(x).mean(-1).mean(-1)  # NCWH -> NC
        return out.view(out.size(0), -1)

# Module to unflatten vectors into images
class Unflatten(torch.nn.Module):
    def __init__(self, in_dim, out_channels, out_height, out_width, activation='leaky_relu'):
        super(Unflatten, self).__init__()
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width
        out_dim = out_channels * out_height * out_width
        linear_layer = torch.nn.Linear(in_dim, out_dim, bias=True)
        self.linear_layer = linear_layer
        self.activation = activation

    def forward(self, x):
        out = import_activation(self.activation)(self.linear_layer(x))
        return out.view([x.shape[0], self.out_channels, self.out_height, self.out_width])


# Convolutional Function
class CNN_Module(torch.nn.Module):
    def __init__(self, in_channel, cnn_layers,obs_shape=64, activation='leaky_relu',with_last_actv=False,name=''):
        super(CNN_Module, self).__init__()

        self.nb_layers = len(cnn_layers)
        self.with_last_actv = with_last_actv
        self.cnn_layers = cnn_layers
        for _ in cnn_layers:
            size_out = conv2d_size_out(obs_shape)
        # In fact thanks to Flatten finalConvSize is not useful
        self.finalConvSize = cnn_layers[-1][-1]*size_out**2
        self.name = name

        layers = []
        for n_lay, (char, scale, out_channel) in enumerate(cnn_layers):
            layer = None
            if (char == '/') or (scale == 1):
                layer = torch.nn.Conv2d(in_channel, out_channel, stride=scale, padding=2, kernel_size=(5, 5), bias=True)
            elif char == '*':
                layer = torch.nn.ConvTranspose2d(in_channel, out_channel, stride=scale, padding=2, kernel_size=(5, 5), output_padding=1, bias=True)
            assert not (layer is None), 'error in char'
            layers.append(layer)
            if activation is not None and (not ((n_lay + 1) == self.nb_layers) or self.with_last_actv):
                layers.append(import_activation(activation))
            in_channel = out_channel

        self.cnn =torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        return out


# Multi-layer perceptron
class MLP_Module(torch.nn.Module):
    def __init__(self, in_dim, MLP_dims, activation='leaky_relu', with_last_actv=False,requires_grad=True,name=''):
        super(MLP_Module, self).__init__()

        self.nb_layers = len(MLP_dims)
        self.MLP_dims = MLP_dims
        self.with_last_actv = with_last_actv
        self.name = name

        layers = []
        for n_lay, out_dim in enumerate(MLP_dims):
            layers.append(torch.nn.Linear(in_dim, out_dim, bias=True))
            if activation is not None and (not ((n_lay + 1) == self.nb_layers) or self.with_last_actv):
                layers.append(import_activation(activation))
            in_dim = out_dim
        self.mlp = torch.nn.Sequential(*layers)


        if not requires_grad:
            for p in self.mlp.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.mlp(x)
        return out


def weight_init_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def weight_init_orthogonal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(m.bias, 0)

def weight_init_random_trunc(m):
    if isinstance(m, torch.nn.Linear):
        std = 0.02
        b = 2 * std
        torch.nn.init.normal_(m.weight, mean=0., std=std).clamp(min=-b,max=b)
        torch.nn.init.constant_(m.bias, 0)

def weight_init_random(m):
    if isinstance(m, torch.nn.Linear):
        std = 0.02
        torch.nn.init.normal_(m.weight, mean=0., std=std)
        torch.nn.init.constant_(m.bias, 0)

def weight_init_high(m):
    if isinstance(m, torch.nn.Linear):
        std = 0.02
        torch.nn.init.normal_(m.weight, mean=0.025, std=std)
        torch.nn.init.constant_(m.bias, 0)

def import_weight_init(weight_init):
    if weight_init == 'orthogonal':
        return weight_init_orthogonal
    elif weight_init == 'random_init':
        return weight_init_random
    elif weight_init == 'random_init_trunc':
        return weight_init_random_trunc
    elif weight_init == 'xavier':
        return weight_init_xavier
    elif weight_init == 'high':
        return weight_init_high


def import_activation(activation,functional=False):
    if functional:
        if activation == 'relu':
            activation = torch.nn.functional.relu
        elif activation == 'elu':
            activation = torch.nn.functional.elu
        elif activation == 'tanh':
            activation = torch.tanh
        elif activation == 'tanh_sigmoid':
            activation = F_tanh_sigmoid
        elif activation == 'leaky_relu':
            activation = F_leaky_relu
    else:
        if activation == 'relu':
            activation = torch.nn.ReLU()
        elif activation == 'elu':
            activation = torch.nn.ELU()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()
        elif activation == 'tanh_sigmoid':
            activation = tanh_sigmoid()
        elif activation == 'leaky_relu':
            activation = torch.nn.LeakyReLU(0.2)
    return activation


def F_leaky_relu(x):
    m = torch.nn.LeakyReLU(0.2)
    activation = lambda i: m(i)
    return activation(x)

class tanh_sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh(x * (2 / 3))

def F_tanh_sigmoid(x):
    return 1.7159 * torch.tanh(x * (2 / 3))

