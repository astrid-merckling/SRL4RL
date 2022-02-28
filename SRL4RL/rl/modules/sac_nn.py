import numpy as np
import torch
import torch.nn as nn

from SRL4RL.utils.nn_torch import MLP_mdn, MLP_Module, import_weight_init, pytorch2numpy
from SRL4RL.utils.utils import get_hidden


class QNetwork(nn.Module):
    def __init__(self, env_params, config):
        super(QNetwork, self).__init__()
        self.method = config["method"]

        nb_hidden = get_hidden(config["nb_hidden"])
        print("nb_hidden critic:", nb_hidden)

        "Q1 & Q2 architectures"
        nb_layer = len(nb_hidden)
        if nb_layer > 1:
            assert not config["linearApprox"], "linearApprox with multiple nb_hidden!"
            activation = config["activation"]
        elif nb_layer == 1:
            if config["linearApprox"]:
                activation = None  # to make a linear policy network
            else:
                activation = config["activation"]

        nb_hidden += [1]
        input_dim = env_params["obs"][0] + env_params["goal"] + env_params["action"]
        self.q1_network = MLP_Module(
            input_dim, nb_hidden, activation=activation, name="Q1"
        )
        self.q2_network = MLP_Module(
            input_dim, nb_hidden, activation=activation, name="Q2"
        )

        if config["weight_init"] != "none":
            weight_init_ = import_weight_init(config["weight_init"])
            self.q1_network.apply(weight_init_)
            self.q2_network.apply(weight_init_)

    def forward(self, action_, x, goal=None):
        if goal is not None:
            x = torch.cat([x, goal], dim=1)
        xu = torch.cat([x, action_], 1)
        x1 = self.q1_network(xu)
        x2 = self.q2_network(xu)
        return x1, x2


def gaussian_logprob(noise, log_sig):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_sig).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def sampleNormal(mu, sig):
    noise = torch.randn_like(mu)
    return mu + noise * sig, noise


class GaussianPolicy(nn.Module):
    def __init__(self, env_params, config):
        super(GaussianPolicy, self).__init__()
        self.method = config["method"]

        self.log_sig_min = -10
        self.log_sig_max = 2

        nb_hidden = get_hidden(config["nb_hidden"])
        nb_layer = len(nb_hidden)
        assert config["cutoff"] < nb_layer, "not enoug layers for cutoff"

        print("nb_hidden actor: {}, cutoff: {}".format(nb_hidden, config["cutoff"]))
        if nb_layer > 1:
            assert not config["linearApprox"], "linearApprox with multiple nb_hidden!"
        elif nb_layer == 1:
            if config["linearApprox"]:
                with_last_actv = False  # to make a linear policy network
            else:
                with_last_actv = True

        input_dim = env_params["obs"][0] + env_params["goal"]
        if config["cutoff"] > 0:
            self.pi_network, self.mean_linear, self.log_sig_linear, _ = MLP_mdn(
                input_dim,
                nb_hidden + [env_params["action"]],
                cutoff=config["cutoff"],
                activation=config["activation"],
            )
        else:
            self.pi_network = MLP_Module(
                input_dim,
                nb_hidden,
                activation=config["activation"],
                with_last_actv=with_last_actv,
                name="Pi",
            )
            self.mean_linear = nn.Linear(nb_hidden[-1], env_params["action"])
            self.log_sig_linear = nn.Linear(nb_hidden[-1], env_params["action"])

        if config["weight_init"] != "none":
            weight_init_ = import_weight_init(config["weight_init"])
            self.pi_network.apply(weight_init_)

        self.action_scale = torch.tensor(
            (env_params["action_max"] - env_params["action_min"]) / 2.0
        )
        self.action_bias = torch.tensor(
            (env_params["action_max"] + env_params["action_min"]) / 2.0
        )

    def forward(self, x, oneHot, goal=None):
        if goal is not None:
            x = torch.cat([x, goal], dim=1)
        x = self.pi_network(x)
        mean = self.mean_linear(x)
        log_sig = self.log_sig_linear(x)
        log_sig = torch.clamp(log_sig, min=self.log_sig_min, max=self.log_sig_max)

        assert not torch.isnan(mean).any().item(), "isnan in mean!!"
        assert not torch.isnan(log_sig).any().item(), "isnan in log_sig!!"
        return mean, log_sig

    def normalizePi(self, mu, pi, log_pi):
        """Apply squashing function.
        See appendix C from https://arxiv.org/pdf/1812.05905.pdf
        """
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        pi = torch.tanh(pi)
        epsilon = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
        log_pi -= torch.log(self.action_scale * (1 - pi.pow(2)) + epsilon).sum(
            -1, keepdim=True
        )
        pi = pi * self.action_scale + self.action_bias
        return mu, pi, log_pi

    def sample(self, state, oneHot=False, goal=None):
        mu, log_sig = self.forward(state, oneHot=oneHot, goal=goal)
        sig = log_sig.exp()
        # for repameterization trick (mu + sig * N(0,1))
        # Pre-squash distribution and sample
        x_t, noise = sampleNormal(mu=mu, sig=sig)
        log_pi = gaussian_logprob(noise, log_sig)
        mu, pi, log_pi = self.normalizePi(mu, x_t, log_pi)

        # Deterministic action
        assert not torch.isnan(pi).any().item(), "isnan in pi!!"
        return pi, log_pi, mu

    def select_actions(self, state, evaluate=False, goal=None):
        if evaluate:
            _, _, pi = self.sample(state, goal=goal)
        else:
            pi, _, _ = self.sample(state, goal=goal)
        # return action.detach().cpu().numpy()[0]
        return pytorch2numpy(pi).squeeze(axis=0)
