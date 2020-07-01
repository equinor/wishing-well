# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:35:52 2020

@author: 2920029
"""
## simple example of how to run MPC ## 
import gym
import numpy as np
import matplotlib.pyplot as plt
import env.AutoDrillEnv
from mpc import solveMPC

env = gym.make('AutoDrillEnv-v0')
mpcPolicy = solveMPC(env)
for i in range(1):
    s = env.reset()
    done = False
    while not done:
    #    Call mpc to solve optimal control problem at current step
        a = mpcPolicy.action(s,plot=True,integer_control=False)
        print('action taken:',a)
        s,r,done,_ = env.step(a)
    simstates = env.render(save=False, show=True)
