from gym_ww.envs.ww3_env import WellPlot3Env

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

class WellPlot5Env(WellPlot3Env):

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.actions_dict = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(-1,1), 4:(1,1)}
        #Actions: 0:Left, 1:Down, 2:Right, 3:Down_left, 4:Down_right