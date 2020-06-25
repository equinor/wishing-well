import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt


class WellPlot3Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(WellPlot3Env, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)             #I think the observation space is just one (current point/state of drill)

        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111)

        self.reward_dict = {}
        self.actions_dict = {0:(-1,0), 1:(0,1), 2:(1,0)}        #Actions: 0:Left, 1:Down, 2:Right

        self.init_state = None
        self.end_state = None
        self.current_state = None
        self.next_state = None

        # Default values
        self.grid_width = 7
        self.grid_height = 7
        self.distance_points = 20
        self.default_reward = 0                      #This is the default reward for an action that has not been assigned another by the agent yet

        init_state = (1, 0)
        end_state = (5,6)

        self.set_init_end_state(init_state,end_state)       #Also makes a call to update_plot


    #Method for setting another init and end state than default
    def set_init_end_state(self,init_state,end_state):
        self.init_state = self.scale_point(init_state)   # starting point
        self.end_state = self.scale_point(end_state)     # ending point
        self.current_state = self.init_state
        self.update_plot()

    #Method for setting grid-size other than default
    def set_grid_size(self,width,height):
        self.grid_width = width
        self.grid_height = height
        self.update_plot()

    #Method for updating plot in case user edits parameters from default
    def update_plot(self):
        self.subplot.cla()
        x = np.arange(0, self.grid_width*self.distance_points, self.distance_points)
        y = np.arange(0, self.grid_height*self.distance_points, self.distance_points)
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
            self.reward_dict[state] = self.default_reward
            return self.default_reward

    #Allows you to set reward for specified state
    def set_reward(self, state, reward):
        self.reward_dict[state] = reward


    def step(self, action):
        action = self.scale_point(self.actions_dict[action])
        self.current_state = (self.current_state[0]+action[0], self.current_state[1]+action[1])
        done = self.current_state == self.end_state
        reward = self.get_reward(self.current_state)
        return self.current_state, reward, done, {}

    #Returns the next step without changing current_state
    def check_step(self,action):
        action = self.scale_point(self.actions_dict[action])
        return (self.current_state[0]+action[0], self.current_state[1]+action[1])


    def reset(self):
        self.current_state = self.init_state
        self.next_state = None
        return self.current_state

    

    #def render(self, mode='human'):
        #Not relevant for this problem scince we cant iteratively render the plot
    

    def plot_line(self,point1,point2):
        plt.plot((point1[0], point2[0]), (point1[1], point2[1]))

    def plot_path(self,path):
        if path is None:
            raise TypeError("Path was of type None")

        for i in range(1,len(path)):
            self.plot_line(path[i-1],path[i])


    def close(self):
        return self.fig


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
#path = []
#path.append(env.init_state)
#path.append(env.step(0)[0])
#path.append(env.step(1)[0])
#path.append(env.step(2)[0])
#print(path)
#env.plot_path(path)
#env.close()