from gym_ww.envs.ww3_env import WellPlot3Env

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

class WellPlot16EnvAdvanced(WellPlot3Env):

    def __init__(self):
        super().__init__()      
        self.action_space = spaces.Discrete(16)
        self.actions_dict ={                                #The actions are sorted as a circle, starting from straight right going downwards
            0:(1,0),                                        #Actions: 0:Right, 
            1:(2,1), 2:(1,1), 3:(1,2),                      #Actions: 1:Down_Right_Right, 2:Down_Right, 3:Down_Down_Right
            4:(0,1),                                        #Actions: 4:Down
            5:(-1,2), 6:(-1,1), 7:(-2,1),                   #Actions: 5:Down_Down_Left, 6:Down_Left, 7:Down_Left_Left
            8:(-1,0),                                       #Actions: 8:Left
            9:(-2,-1), 10:(-1,-1), 11:(-1,-2),              #Actions: 9:Up_Left_Left, 10:Up_Left, 11:Up_Up_Left
            12:(0,-1),                                      #Actions: 12:Up
            13:(1,-2), 14:(1,-1), 15:(2,-1)                 #Actions: 13:Up_Up_Right, 14:Up_Right, 15:Up_Right_Right
        }



