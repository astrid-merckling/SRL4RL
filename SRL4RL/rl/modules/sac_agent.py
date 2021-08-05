import torch
import numpy as np
from SRL4RL.utils.nn_torch import numpy2pytorch, pytorch2numpy

from SRL4RL.rl.utils.mpi_utils import sync_networks, sync_params
from SRL4RL.rl.modules.agent import Agent
from SRL4RL.rl.modules.agent_utils import loadPi
from SRL4RL.rl.modules.sac_nn import QNetwork, GaussianPolicy

"""
SAC (MPI-version)
"""

class sac_agent(Agent):
    def __init__(self, config, env, env_params, runner):
        # create the network
        if config['dir']:
            suffix = '_last' if 'last_eval' in config else '_best'
            config['o_mean'], config['o_std'], config['g_mean'], config[
                'g_std'], self.actor_network, self.critic_network = loadPi(config['dir'], model_type = 'model' + suffix, withQ=True)
            self.actor_network.train(), self.critic_network.train()
        else:
            self.actor_network = GaussianPolicy(env_params, config)
            self.critic_network = QNetwork(env_params, config)
        super(sac_agent, self).__init__(config, env, env_params, runner)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.critic_target_network = QNetwork(env_params, config)
        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        for p in self.critic_target_network.parameters():
            p.requires_grad = False

        self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=config['lr_actor'])
        self.critic_network_optim = torch.optim.Adam(self.critic_network.parameters(), lr=config['lr_critic'])

        device = self.cpu
        self.to_device(device=device)
        # Target Entropy
        self.automatic_entropy_tuning = config['automatic_entropy_tuning'] if self.agent == 'SAC' else False
        if self.automatic_entropy_tuning:
            self.target_entropy = np.float32(-self.env_params['action'])
            self.log_alpha = numpy2pytorch(np.array(np.log(config['init_temperature'])), differentiable=True,
                                           device=device)
            self.alpha = self.log_alpha.exp()
            sync_params(self.log_alpha)
            self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=config['lr_alpha'])

        self.train(training=False)
        self.entropyPi_sum = 0
        self.alpha_sum = 0

    # update the network
    def update_network(self, update_actor_and_alpha=True):

        # sample the episodes
        transitions = self.buffer.sample(self.batch_size)
        # do the normalization
        device = self.cpu
        inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor, g_tensor, not_done_tensor = self.prepare_data(
            transitions, device=device)
        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            actions_next, log_pi_next, _ = self.actor_network.sample(inputs_next_norm_tensor, oneHot=True,
                                                                     goal=g_tensor)
            # Target Q-values
            qf1_next_target, qf2_next_target = self.critic_target_network(actions_next, inputs_next_norm_tensor,
                                                                          goal=g_tensor)
            # Compute the target Q value: min over all critic_networks targets and add entropy term
            min_qf_next = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * log_pi_next
            # td error + entropy term
            qf_target = r_tensor + not_done_tensor * self.gamma * min_qf_next

        # Get current Q estimates for each critic_network network
        # using action from the replay buffer
        qf1, qf2 = self.critic_network(actions_tensor, inputs_norm_tensor, goal=g_tensor)
        # Compute critic_network loss: MSE loss against Bellman backup
        critic_network_loss = torch.nn.functional.mse_loss(qf1, qf_target) + torch.nn.functional.mse_loss(qf2, qf_target)

        """
        Update networks
        """
        # update the qf1 critic_network_network
        # Optimize the critic_network
        self.critic_network_optim.zero_grad()
        critic_network_loss.backward()
        # sync_grads(self.critic_network)
        self.critic_network_optim.step()

        if update_actor_and_alpha:
            # Compute actor_network loss
            # Mean over all critic_network networks
            pi, log_pi, _ = self.actor_network.sample(inputs_norm_tensor, oneHot=True, goal=g_tensor)
            qf1_pi, qf2_pi = self.critic_network(pi, inputs_norm_tensor, goal=g_tensor)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            # Entropy-regularized policy loss
            policy_loss = (self.alpha.detach() * log_pi - min_qf_pi).mean()
            self.entropyPi_sum -= pytorch2numpy(log_pi.mean())
            self.alpha_sum += pytorch2numpy(self.alpha)
            """
              Update networks
            """
            # Optimize the actor_network
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
            if self.automatic_entropy_tuning:
                self.log_alpha_optim.zero_grad()
                alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
                alpha_loss.backward()
                self.log_alpha_optim.step()
                self.alpha = self.log_alpha.exp()
