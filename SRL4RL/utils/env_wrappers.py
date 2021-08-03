import gym
import numpy as np

from SRL4RL.utils.utilsEnv import renderPybullet
from SRL4RL.utils.utils import state_baselines


class GoalWrapper(gym.Wrapper):
    def __init__(self,env):
        gym.Wrapper.__init__(self, env)
        self.new_goal = None

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def compute_reward(self,achieved_goal=None, desired_goal=None,action=None):
        return self.env.reward_reach(achieved_goal, desired_goal)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        ag = self.env.object
        dg = self.env.target
        obs = {'observation': obs,
               'achieved_goal': ag,
               'desired_goal': dg,
               }
        reward = self.compute_reward(achieved_goal=obs['achieved_goal'], desired_goal=obs['desired_goal'], action=action)
        _is_success = (reward+1)
        info = {'is_success': _is_success,}
        return obs, reward, done, info

    def reset(self,keep_goal=False):
        obs = self.env.reset()
        ag = self.env.object
        dg = self.env.target
        if not keep_goal:
            self.new_goal = dg.copy()
        else:
            self.define_goal(self.new_goal)
        obs = {'observation': obs,
               'achieved_goal': ag,
               'desired_goal': self.new_goal,
               }
        return obs

    def define_goal(self,value):
        return self.env.define_goal(value)


class BulletWrapper(gym.Wrapper):
    def __init__(self,env,config):

        gym.Wrapper.__init__(self, env)
        self.method = config['method']
        self.image_size = config['image_size']
        self.color = config['color']
        self.fpv = config['fpv']
        if self.method in state_baselines:
            self.with_images = False
        else:
            self.with_images = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.method == 'pure_noise':
            obs = np.random.normal(0, 1, (obs.shape)).astype(np.float32)
        elif self.method == 'position':
            obs = self.env.robot.measurement()
        elif self.method == 'openLoop':
            obs = np.array([self.env.envStepCounter // self.env.actionRepeat])
        elif self.with_images:
            obs = renderPybullet(self.env, self.__dict__)
        return obs, reward, done, info

    
    def reset(self):
        obs = self.env.reset()
        if self.method == 'pure_noise':
            obs = np.random.normal(0, 1, (obs.shape)).astype(np.float32)
        elif self.method == 'position':
            obs = self.env.robot.measurement()
        elif self.method == 'openLoop':
            obs = np.array([self.env.envStepCounter // self.env.actionRepeat])
        elif self.with_images:
            obs = renderPybullet(self.env, self.__dict__)
        return obs        
