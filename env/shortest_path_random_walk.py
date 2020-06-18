# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 08:48:53 2020

@author: Jakob Torben
"""

import numpy as np
import matplotlib.pyplot as plt
import random


class shortest_path:
    """
    Class that finds shortest path between points start and stop using several
    random walks and finding the shortest path
    """
    
    def __init__(self, L, step, start, stop):
        self.L = L    # size of square grid
        self.step = step    # distance between points
        self.start = np.array(start)*step    # starting point
        self.stop = np.array(stop)*step    # ending point


    def grid_gen(self):
        """
        Generates a gridpoints and plots the grid.
        Only for visualisation, not used for shortest path.
        """
        
        x = np.arange(0, self.L*self.step, self.step)
        y = np.arange(0, self.L*self.step, self.step)
        X, Y = np.meshgrid(x, y)
        
        plt.plot(X, Y, 'ob')
        plt.gca().invert_yaxis()
        plt.show()
        
        
    def action(self):
        """
        Controls the action of the agent using a random walk in the x and y 
        direction.
        """
        
        valid = False
        while valid == False:
            x = random.choice([-self.step, 0, self.step])
            y = random.choice([-self.step, 0, self.step])
            
            # ensures that the random step is within the grid
            if self.pos[0] + x >= 0 and self.pos[0] + x < self.step*self.L and \
                self.pos[1] + y >= 0 and self.pos[1] + y < self.step*self.L:
                valid = True
                
        self.pos[0] += x
        self.pos[1] += y
               
        
    def game(self):
        """
        A single game where the agent starts at the starting point and performs
        a random walk until it reaches the ending point. Its path is saved.
        """
        
        max_it = 10000
        self.iteration = 0
        self.pos = list(self.start)
        path = [self.pos[:]]

        # take iteration number of random steps to reach ending point
        while self.iteration <= max_it:
            self.action()
            self.iteration += 1
            path.append(self.pos[:])
            
            if np.all(self.pos == self.stop):
                self.policy = np.array(path)
                print("End position found in ", self.iteration, "iterations")
                break

            if self.iteration == max_it:
                print("Shortest path not found in ", max_it, " iterations")
        
        
    def optimal_path(self, games):
        """
        Finds the shortest path by running games number of games and finding 
        the shortest path.
        """ 
        
        valuefunc = np.zeros(games)
        policies = []
        
        for i in range(games):
            self.game()
            # negative so that valuefunc can be maximised
            valuefunc[i] = -self.iteration    
            policies.append(self.policy)
            
        index = np.where(valuefunc == max(valuefunc) )[0][0]  # finds max of valuefunc
        shortest_path = policies[index]
        # print("Shortest path: ", shortest_path)
        print("Path length: ", -max(valuefunc))
        
        plt.plot(self.start[0], self.start[1], "or", label="start", markersize=10)
        plt.plot(self.stop[0], self.stop[1], "og", label="stop", markersize=10)
        x, y = shortest_path.T
        plt.plot(x, y, label="optimal path")
        plt.legend()
        

model = shortest_path(7, 30, [1, 0], [5, 6])
    
model.grid_gen()
model.optimal_path(100)  