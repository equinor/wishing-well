# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 08:48:53 2020

@author: Jakob Torben
"""

import numpy as np
import matplotlib.pyplot as plt
import random


class shortest_path:
    
    def __init__(self, L, step, start, stop):
        self.L = L
        self.step = step
        self.start = np.array(start)*step
        self.stop = np.array(stop)*step


    def grid_gen(self):
        x = np.arange(0, self.L*self.step, self.step)
        y = np.arange(0, self.L*self.step, self.step)
        X, Y = np.meshgrid(x, y)
        
        plt.plot(X, Y, 'ob')
        plt.gca().invert_yaxis()
        plt.show()
        
        
    def action(self):
        
        valid = False
        while valid == False:
            x = random.choice([-self.step, 0, self.step])
            y = random.choice([-self.step, 0, self.step])
            
            if self.pos[0] + x >= 0 and self.pos[0] + x < self.step*self.L and \
                self.pos[1] + y >= 0 and self.pos[1] + y < self.step*self.L:
                valid = True
                
        self.pos[0] += x
        self.pos[1] += y
               
        
        
    def game(self):
        
        max_it = 10000
        self.iteration = 0
        path = np.zeros([10000, 2])
        # self.pos = self.start
        self.pos = [30, 0]  # manually set start, need to find bug
        path[self.iteration, :] = self.pos

        while self.iteration <= max_it:
            self.action()
            self.iteration += 1
            path[self.iteration, :] = self.pos
            
            if np.all(self.pos == self.stop):
                self.path = path[0:self.iteration+1]
                print("End position found in ", self.iteration, "iterations")
                break

            if self.iteration == max_it:
                print("Shortest path not found in ", max_it, " iterations")
        
        
    def max_pi(self, games):
        
        valuefunc = np.zeros(games)
        policies = []
        
        for i in range(games):
            self.pos = [30, 0]
            path = []
            self.game()
            valuefunc[i] = -self.iteration
            policies.append(self.path)
            
        index = np.where(valuefunc == max(valuefunc) )[0][0]
        optimal_path = policies[index]
        # print("Shortest path: ", optimal_path)
        print("Path length: ", -max(valuefunc))
        
        plt.plot(self.start[0], self.start[1], "or", label="start", markersize=10)
        plt.plot(5*30, 6*30, "og", label="stop", markersize=10)    # need to find bug on self.stop
        x, y = optimal_path.T
        plt.plot(x, y, label="optimal path")
        plt.legend()
        