# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:04:12 2020

@author: 2920029
"""
import gym
import env.smallVaringGoalRep1Environment
from stable_baselines import DQN

env = gym.make('SmallVaringGoalRep1Env-v0')

from stable_baselines.deepq.policies import MlpPolicy
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)


# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
