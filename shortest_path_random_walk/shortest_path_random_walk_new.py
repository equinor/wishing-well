# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 08:48:53 2020

@author: Jakob Torben
"""

import matplotlib as mpl            #Import entire matplotlib

import matplotlib.pyplot as plt     #Still use plt for pyplot for familiarity
import numpy as np
import random
import gym

#from gym import envs
from gym_ww import envs

#gym.make('well-plot-3-v0')

class shortest_path:
    def __init__(self):
        self.env = gym.make('well-plot-5-v0')
        self.policy = []

        self.num_tries = 15
        self.max_len_path = 10
    
    def game(self):
        self.policy = [self.env.state]
        for i in range(self.max_len_path):

            if not self.possible_to_move(self.env.state):
                break

            action = self.env.action_space.sample()
            while not self.valid_action(action):                        #Keep choosing random action until it finds valid action
                action = self.env.action_space.sample()

            new_state, reward, done, info = self.env.step(action)
            self.policy.append(new_state)

            if done:
                print("Found path of length ", i+1)
                return self.policy
                
        print("Did not find path shorter than", self.max_len_path)
        return None
        

    def optimal_path(self, policies):
        policies = []
        
        for i in range(self.num_tries):
            policy = self.game()
            if not policy == None:
                policies.append(policy)
            self.env.reset()

        policies = [ list(policy) for policy in policies ]
        policies.sort(key=len)
        if not len(policies) == 0:
            shortest_policy = policies[0]
            print("Length of shortest path found: ", len(shortest_policy)-1)
            self.env.plot_path(shortest_policy)
        else:
            print("No path of length ", self.max_len_path," or shorter found.")


    #Check if action is in board. Can be replaced by negative rewards when using RL: env.set_reward(state,reward)
    def valid_action(self,action):
        new_state = self.env.check_step(action)
        valid_state = self.env.valid_state(new_state)
        policy_list = [ list(policy) for policy in self.policy ]
        new_state_list = list(new_state)
        not_prev_state = new_state_list not in policy_list
        return valid_state and not_prev_state

    def possible_to_move(self,state):
        for action in range(self.env.action_space.n):
            new_state = self.env.check_step(action)
            policy_list = [ list(policy) for policy in self.policy ]
            new_state_list = list(new_state)
            valid_state = self.env.valid_state(new_state) and new_state_list not in policy_list
            if valid_state:
                return True
        return False
    
    def get_figure(self):
        return self.env.fig
