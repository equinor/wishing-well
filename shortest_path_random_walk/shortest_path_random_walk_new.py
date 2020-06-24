# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 08:48:53 2020

@author: Jakob Torben
"""

import matplotlib as mpl            #Import entire matplotlib and use WebAgg for showing in browser
#from matplotlib.figure import Figure

import matplotlib.pyplot as plt     #Still use plt for pyplot for familiarity
import numpy as np
import random
import gym

#from gym import envs
from gym_ww import envs


#all_envs = envs.registry.all()
#env_ids = [env_spec.id for env_spec in all_envs]
#print(env_ids)

#gym.make('well-plot-3-v0')

class shortest_path:
    def __init__(self):
        self.env = gym.make('well-plot-5-v0')
        self.iteration = 0
        self.policy = []
        self.policies = []

    
    def game(self):
        max_it = 10000
        policy = [self.env.current_state]

        # take iteration number of random steps to reach ending point
        while self.iteration <= max_it:
            action = self.env.action_space.sample()

            #while not self.valid_action(action):                        #Keep choosing random action until it finds valid action
            #    action = self.env.action_space.sample()

            new_state, reward, done, info = self.env.step(action)
            self.iteration += 1
            policy.append(new_state)
            
            if done:
                self.policy = policy
                print("End position found in ", self.iteration, "iterations")
                break

            if self.iteration == max_it:
                print("Shortest path not found in ", max_it, " iterations")


    def optimal_path(self):
        num_games = 100
        valuefunc = np.zeros(num_games)
        
        #for i in range(num_games):
        #    self.game()
        #    self.env.reset()
        #
        #    valuefunc[i] = -self.iteration
        #    self.policies.append(self.policy)
        self.policy.append(self.env.init_state)
        self.policy.append(self.env.step(1)[0])
        self.policy.append(self.env.step(1)[0])
        self.policy.append(self.env.step(1)[0])
        self.policy.append(self.env.step(1)[0])
        self.policy.append(self.env.step(1)[0])
        self.policies.append(self.policy)

        #index = np.where(valuefunc == max(valuefunc) )[0][0]
        index = 0
        shortest_path = self.policies[index]
        #print("Path length: ", -max(valuefunc))
        self.env.plot_path(shortest_path)

    #Check if action is in board. Can be replaced by negative rewards when using RL: env.set_reward(state,reward)
    def valid_action(self,action):
        new_state = self.env.check_step(action)
        return (0 <= new_state[0] <= self.env.grid_width*self.env.distance_points) and (0 <= new_state[1] <= self.env.grid_height*self.env.distance_points)


    def get_figure(self):
        return self.env.fig
