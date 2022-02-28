import os

import gym
import numpy as np
import torch
from bullet_envs.utils import AddNoise, env_with_goals

from SRL4RL.utils.nn_torch import CNN, numpy2pytorch, pytorch2numpy, save_model
from SRL4RL.utils.utilsEnv import add_noise

np2torch = lambda x, device="cpu": numpy2pytorch(x, differentiable=False, device=device)
CPU = torch.device("cpu")


def last_layer(x):
    return torch.sigmoid(x)


class StateWrapper(gym.Wrapper):
    ## Wrapper to reset runner after env is reset
    def __init__(self, env, runner, demo, device=CPU):
        gym.Wrapper.__init__(self, env)
        self.device = torch.device(device)
        self.demo = demo
        self.runner = runner
        self.method = runner.method
        self.n_stack = runner.n_stack
        self.nc = 3 if runner.color else 1
        self.actionRepeat = self.runner.actionRepeat if self.n_stack > 1 else 1

        obs = self.env.reset()
        self.obs_shape = list(obs.shape)
        self.obs_shape[1] *= self.n_stack

        self.rewardFactor = (
            1 if self.runner.env_name in env_with_goals else self.actionRepeat
        )

    def reset(self):
        current_obs = self.env.reset()
        if self.n_stack > 1:
            obs = np.zeros((self.obs_shape), np.float32)
            for step_rep in range(self.n_stack):
                obs[:, step_rep * self.nc : (step_rep + 1) * self.nc] = current_obs
            if self.n_stack > 1 and self.runner.actionRepeat == 1:
                "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
                self.obs = obs
        else:
            obs = current_obs

        obs = numpy2pytorch(obs, differentiable=False, device=self.device)
        if self.runner.with_nextObs:
            self.obs_past = obs

        self.runner.resetState()  # before update_state()
        state = self.runner.update_state(obs, self.demo)
        if self.runner.stack_state:
            state = torch.cat((state, state), dim=1)
            self.old_state = self.runner.state
        elif self.runner.double_state:
            state = torch.cat((state, state), dim=1)
        state = pytorch2numpy(state.squeeze())
        return state

    def step(self, action):
        if self.n_stack > 1:
            obs = np.zeros((self.obs_shape), np.float32)
        for step_rep in range(self.actionRepeat):
            current_obs, r, done, info = self.env.step(action)
            if self.n_stack > 1 and self.runner.actionRepeat == 1:
                "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
                self.obs[:, -2 * self.nc :] = self.obs[:, : 2 * self.nc]
                self.obs[:, : self.nc] = current_obs
                obs = self.obs
            elif self.n_stack > 1:
                if (step_rep + 1) > (self.actionRepeat - self.n_stack):
                    obs[
                        :, (step_rep - 1) * self.nc : ((step_rep - 1) + 1) * self.nc
                    ] = current_obs
            elif (step_rep + 1) == self.actionRepeat:
                assert step_rep < 2, "actionRepeat is already performed in env"
                obs = current_obs

        obs = numpy2pytorch(obs, differentiable=False, device=self.device)
        if self.runner.with_nextObs:
            state = self.runner.update_state(self.obs_past, self.demo)
            self.obs_past = obs
        else:
            state = self.runner.update_state(obs, self.demo)

        if self.runner.stack_state:
            state = torch.cat((state, self.old_state), dim=1)
            self.old_state = self.runner.state
        elif self.runner.double_state:
            state = torch.cat((state, state), dim=1)
        state = pytorch2numpy(state.squeeze())
        return state, r * self.rewardFactor, done, info


class Runner:
    def __init__(self):
        pass

    def forward(self, actor, x, evaluate=False, goal=None, random=False, env=None):
        if random:
            self.pi = env.action_space.sample()
        else:
            with torch.no_grad():
                self.pi = actor.select_actions(x, evaluate=evaluate, goal=goal)
        return self.pi

    def resetState(self):
        pass


class StateRunner(Runner):
    def __init__(self, config):
        super().__init__()
        if config:
            config["srl_early_stop"] = False
        self.color = config["color"]
        self.n_stack = config["n_stack"]
        self.stack_state = config["stack_state"]
        if "double_state" in config:
            self.double_state = config["double_state"]
        else:
            self.double_state = False
        self.debug = config["debug"]
        self.env_name = config["env_name"]
        self.actionRepeat = config["actionRepeat"]
        self.image_size = config["image_size"]
        self.fpv = config["fpv"]
        self.method = config["method"]
        if "SRL_name" in config:
            self.SRL_name = config["SRL_name"]
        self.device = config["device"]
        self.save_dir = config["save_dir"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.noise_type = config["noise_type"]
        self.elapsed_gradients = (
            self.elapsed_epochs
        ) = self.elapsed_time = self.elapsed_steps = 0

        self.with_nextObs = config["method"] in ["XSRL"]
        self.with_noise = self.noise_type != "none"
        if self.with_noise:
            self.noise_adder = AddNoise(config)
        else:
            self.noise_adder = None

        self.noiseParams = {
            "with_noise": self.with_noise,
            "flickering": config["flickering"],
        }

    def initState(self):
        return numpy2pytorch(
            np.random.normal(0, 0.02, self.state_dim),
            differentiable=False,
            device=self.device,
        ).unsqueeze(0)

    def resetState(self):
        self.old_state = self.initState()

    def update_state(self, x, demo=False):
        with torch.no_grad():
            inputs = add_noise(x, self.noise_adder, self.noiseParams)
            self.state = self.encoder(inputs.to(self.device))
        if demo:
            "save last inputs to record video"
            self.last_inputs = pytorch2numpy(inputs)[0][-3:, :, :].transpose(1, 2, 0)
        return self.state


class RandomNetworkRunner(StateRunner):
    def __init__(self, n_channels, config):
        super().__init__(config)

        if config["srl_path"]:
            self.encoder = torch.load(
                os.path.join(config["srl_path"], "state_model.pt"),
                map_location=torch.device("cpu"),
            )
        elif config["my_dir"]:
            self.encoder = torch.load(
                os.path.join(config["save_dir"], "state_model.pt"),
                map_location=torch.device("cpu"),
            )
        else:
            self.encoder = CNN(
                config["state_dim"],
                n_channels,
                activation=config["activation"],
                debug=config["debug"],
            )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def save_state_model(self, save_path):
        print("Saving models ......")
        save_model(self.encoder, save_path + "state_model")

    def train(self, training=True):
        print("random_nn cannot be trained")
        pass

    def to_device(self, device="cpu"):
        self.encoder.to(device)
