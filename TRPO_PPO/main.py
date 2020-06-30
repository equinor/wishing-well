# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:43:41 2020

@author: 2920029
"""

import gym
import env.AutoDrillEnv
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy


env = gym.make('AutoDrillEnv-v0')
######## use TRPO
model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100)

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
    
#### use PPO
from stable_baselines import PPO1

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()