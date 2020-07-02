import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

#gym.logger.set_level(40)

class WellPlot3Env(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(WellPlot3Env, self).__init__()

        self.fig = plt.figure()
        self.subplot = self.fig.add_subplot(111)

        self.reward_dict = {}
        self.actions_dict = {0:(-1,0), 1:(0,1), 2:(1,0)}        #Actions: 0:Left, 1:Down, 2:Right

        # Default values
        self.grid_width = 7
        self.grid_height = 7
        self.distance_points = 20
        self.default_reward = 0                      #This is the default reward for an action that has not been assigned another by the agent yet

        self.init_state = self.scale_point([1, 0])
        self.end_state = self.scale_point([5,6])
        self.state = self.init_state

        self.action_space = spaces.Discrete(3)
        self.update_obv_space()
        self.update_plot()
        
        self.seed()

    ###################### SETUP HELP METHODS BELOW ########################

    def update_obv_space(self):                                          #Separated from __init__ to be updateable
        low = np.array([0,0],dtype=np.int32)                             #The lowest possible x,y value is [0,0]
        
        max_width = (self.grid_width-1)*self.distance_points
        max_height = (self.grid_height-1)*self.distance_points
        high = np.array([max_width,max_height],dtype=np.int32)           #The highest possible x,y value is [width,height] of board

        self.observation_space = spaces.Box(low,high,dtype=np.int32)


    #Method for setting another init and end state than default
    def set_init_end_state(self,init_state,end_state):
        self.init_state = self.scale_point(init_state)   # starting point
        self.end_state = self.scale_point(end_state)     # ending point
        self.state = self.init_state
        self.update_plot()

    #Method for setting grid-size other than default
    def set_grid_size(self,width,height,distance_points=20):
        self.grid_width = width
        self.grid_height = height
        self.distance_points = distance_points
        self.update_obv_space()
        self.update_plot()

    #Method for updating plot in case user edits default values (ex: size,init/end state)
    def update_plot(self):
        self.subplot.cla()
        x = np.arange(0, self.grid_width*self.distance_points, self.distance_points)
        y = np.arange(0, self.grid_height*self.distance_points, self.distance_points)
        X, Y = np.meshgrid(x, y)
        
        self.subplot.plot(X, Y, 'ob',markersize=2)
        self.fig.gca().invert_yaxis()

        self.subplot.plot(self.init_state[0], self.init_state[1], "or", label="start", markersize=10)
        self.subplot.plot(self.end_state[0], self.end_state[1], "og", label="stop", markersize=10)
        self.subplot.legend()

    #Scales point to fit grid
    def scale_point(self,point):
        return np.array([point[0]*self.distance_points, point[1]*self.distance_points])


    ##################### OPENAI GYM ENV METHODS BELOW #######################

    def step(self, action):
        action = self.scale_point(self.actions_dict[action])
        self.state = np.array([self.state[0]+action[0], self.state[1]+action[1]])
        self.state = np.clip(self.state,0,(self.grid_height-1)*self.distance_points)  #Makes the state stay inside the boundary
        done = np.array_equal(self.state,self.end_state)
        reward = self.get_reward(self.state)
        return self.state, reward, done, {}

    def seed(self, seed=None):                                  #Seed method for generating something random? Not relevant for our problem
        self.np_random, seed = seeding.np_random(seed) 
        return [seed]

    def render(self, path_x,path_y):

        plt.xlim([0,(self.grid_width-1)*self.distance_points])
        plt.ylim([(self.grid_height-1)*self.distance_points,0])

        plt.xlabel('Horizontal') 
        plt.ylabel('Depth')
        self.subplot.plot(path_x,path_y)
        self.subplot.grid()
        self.subplot.set_axisbelow(True)
        
        return self.fig


    def plot_line(self,point1,point2):
        plt.plot((point1[0], point2[0]), (point1[1], point2[1]))

    def plot_path(self,path):
        if path is None:
            raise TypeError("Path was of type None")

        for i in range(1,len(path)):
            self.plot_line(path[:,0][i],path[:,1][i])





        #Plot region to avoid
#        circle = plt.Circle((self.c_y,-self.c_x),self.c_r,color='C2')
#        ax.add_artist(circle)

    def reset(self):
        self.state = self.init_state
        return self.state

    def close(self):
        return self.fig

    
    ########################### BONUS METHODS BELOW ###########################

    #Returns the next step without changing state
    def check_step(self,action):
        action = self.scale_point(self.actions_dict[action])
        return np.array([self.state[0]+action[0], self.state[1]+action[1]])
    
    def valid_state(self,state):
        return (0 <= state[0] <= (self.grid_width-1)*self.distance_points) and (0 <= state[1] <= (self.grid_height-1)*self.distance_points)
    
    #Allows you to get reward for specified state
    def get_reward(self,state):
        state = tuple(state)
        if self.valid_state(state)==0:
            return -1
        if state in self.reward_dict:
            return self.reward_dict[state]
        else:
            self.reward_dict[state] = self.default_reward
            return self.default_reward

    #Allows you to set reward for specified state
    def set_reward(self, state, reward):
        state = tuple(state)
        self.reward_dict[state] = reward


    def plot_line(self,point1,point2):
        plt.plot((point1[0], point2[0]), (point1[1], point2[1]))

    def plot_path(self,path):
        if path is None:
            raise TypeError("Path was of type None")

        for i in range(1,len(path)):
            self.plot_line(path[i-1],path[i])
    
    def plot_point(self,point):
        self.subplot.plot(point[0], point[1], "or")