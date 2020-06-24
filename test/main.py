# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:04:12 2020

@author: 2920029
"""

import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import env.smallVaringGoalRep1Environment
env = gym.make('SmallVaringGoalRep1Env-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()