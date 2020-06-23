import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

#import webbrowser


class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box()

        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111)

        self.reward_dict = {}
        self.actions_dict = {0:(-1,0), 1:(0,1), 2:(1,0)}

        # Default values
        self.grid_size = 7
        self.distance_points = 20

        init_state = (1, 0)
        end_state = (5,6)

        self.init_state = None
        self.end_state = None
        self.current_state = None
        self.next_state = None

        self.set_init_end_state(init_state,end_state)       #Also makes a call to update_plot


    #Method for setting another init and end state than default
    def set_init_end_state(self,init_state,end_state):
        self.init_state = self.scale_point(init_state)   # starting point
        self.end_state = self.scale_point(end_state)     # ending point
        self.current_state = self.init_state
        self.update_plot()

    #Method for setting grid-size other than default
    def set_grid_size(self,size):
        self.grid_size = size
        self.update_plot()

    #Method for updating plot in case user edits parameters from default
    def update_plot(self):
        self.subplot.cla()
        x = np.arange(0, self.grid_size*self.distance_points, self.distance_points)
        y = np.arange(0, self.grid_size*self.distance_points, self.distance_points)
        X, Y = np.meshgrid(x, y)
        
        self.subplot.plot(X, Y, 'ob')
        self.fig.gca().invert_yaxis()

        self.subplot.plot(self.init_state[0], self.init_state[1], "or", label="start", markersize=10)
        self.subplot.plot(self.end_state[0], self.end_state[1], "og", label="stop", markersize=10)
        self.subplot.legend()

    #Scales point to fit grid
    def scale_point(self,point):
        return (point[0]*self.distance_points, point[1]*self.distance_points)


    #Allows you to get reward for specified state
    def get_reward(self,state):
        if state in self.reward_dict:
            return self.reward_dict[state]
        else:
            self.reward_dict[state] = 0
            return 0

    #Allows you to set reward for specified state
    def set_reward(self, state, reward):
        self.reward_dict[state] = reward


    #Actions: 0:Left, 1:Down, 2:Right
    def step(self, action):
        done = False
        action = self.scale_point(self.actions_dict[action])

        self.current_state = (self.current_state[0]+action[0], self.current_state[1]+action[1])

        if self.current_state == self.end_state:
            done = True

        reward = self.get_reward(self.current_state)
            
        return self.current_state, reward, done, {}



    def reset(self):
        self.current_state = self.init_state
        self.next_state = None
        return self.current_state

    

    #def render(self, mode='human'):
        #Not relevant for this problem scince 
    

    def plot_line(self,point1,point2):
        plt.plot((point1[0], point2[0]), (point1[1], point2[1]))

    def plot_path(self,path):
        for i in range(1,len(path)):
            self.plot_line(path[i-1],path[i])


    def close(self):
        plt.show()
        #webbrowser.open('http://localhost:8080')  # Go to example.com


#import gym
#env = gym.make('CartPole-v1')
#env = FooEnv()
#env.reset()
#for _ in range(1000):
#    env.render()
#    env.step(env.action_space.sample())

#init_state = (1, 0)
#end_state = (5, 6)
#env.set_init_end_state(init_state,end_state)

#env.set_grid_size(10)
#env.set_init_end_state((1, 0),(5, 9))
'''
path = []
path.append(env.init_state)
path.append(env.step(0)[0])
path.append(env.step(1)[0])
path.append(env.step(2)[0])
print(path)
env.plot_path(path)
env.close()
'''