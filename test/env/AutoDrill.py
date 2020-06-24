import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
        
class AutoDrill(Env):     

    simstates = np.zeros((1,4)) #Continuous state variables
    
    def __init__(self):
        # 100*90*50 states: Position grid of 100 blocks ; angles -180 to 180 at 4deg precision ; build rate -25 to 25 at 1deg/ft precision
        # 21 actions: -1 to 1 duty cycle in .1 resolution (Where duty cycle is the force of input)
        self.sD = [100,100,91,51] #Dimensions of state space
        self.state_matrix = np.zeros(self.sD) #Gridded states represented in 4-D matrix form
        self.state_array = np.zeros(np.product(self.sD))
        self.X = 1000 #Size in feet of position grid x perimeter
        self.Y = 500 #Size in feet of position grid y perimeter
        
        nS=np.product(self.sD)
        env_spec = EnvSpec(nS, 21, 1.) #nS,nA,gamma
        super().__init__(env_spec)
        self.tile_width = np.array((50,5,5,10))

        
        #Drilling state bounds:
        self.stateLow = np.array([0., 0., -10., -25.])
        self.stateHigh = np.array([1000, 500, 100, 25])
            
        self.numActions = 21
        
    def awhere(self,ndloc,ndsize):
        pos = 0
        for p in range(ndloc.shape[0]-1):
            pos += ndloc[p]*np.product(ndsize[p+1:])
        pos += ndloc[-1]
        return pos
    
    def reset(self):
        global simstates
        #Random Model Parameters
        import random
        self.tau = 15#random.uniform(15.-5,15.+5)
        self.ku = 18#random.uniform(18.-4,18+4)
        
        #Initialize states at zero
#        self.state = np.zeros(4,dtype=int) #Start at home
#        simstates = np.zeros((1,4))
        
        #Initialize states from random distribution
        yo = 0#random.uniform(90-20,90+20)
        xo = 0#random.uniform(0,20)
        
        #Initialize states from array of possible values
#        action= np.load('stacked_stoch_trained_states.npy')
#        so = action[np.random.choice(np.arange(0,action.shape[0],1))]
#        self.state = so
#        simstates = so.reshape(1,-1)

        self.state = np.array([xo,yo,0,0])
        simstates = np.array([[xo,yo,0,0]])
        return self.state
    
    def cont2grid(self, _simstates):
        #Very rudimentary tile coding method. Only action single tiling is used.
        
        x= _simstates[0,0]
        y= _simstates[0,1]
        theta = _simstates[0,2]
        thetadot = _simstates[0,3]
        
        #Determine coarse grid values
        gridx = int(np.round(x*(self.sD[0]-1)/self.X))
        gridy = int(np.round(y*(self.sD[1]-1)/self.Y))
        
        if theta<-180:
            theta=theta+180
        if theta>180: 
            theta=theta-180
        
        gridtheta = int(np.round(theta*(self.sD[2]-1)/2/180)+45)
        gridthetadot = int(np.round(thetadot*(self.sD[2]-1)/2/25)+25)    
        
        #Force limits on states (ie. if x_t+1 < 0, x_t+1 = 0)
        gridx = int(np.clip(gridx,0,self.sD[0]-1))
        gridy = int(np.clip(gridy,0,self.sD[1]-1))
        
        return (gridx,gridy,gridtheta,gridthetadot) 
        
    def disp_traj(self, ep_n, save_flag):
        fig = plt.figure(1)
        plt.plot(simstates[:,1],-simstates[:,0])
        plt.xlim([0,self.Y])
        plt.ylim([-self.X,0])
        plt.xlabel('Cross Section ft')
        plt.ylabel('TVD ft')
        fig = plt.figure(1)
        if save_flag == 1:
            fig.savefig('Episode_'+str(ep_n)+'.png')

#        plt.show()

    def step(self, action):
        global simstates
        # assert action in range(self.spec.nA), "Invalid Action"
        
        #Model Parameters
        dist=0
        tau=self.tau
        ku=self.ku

        #Soft Rock Layer
#        layer_size = 200
#        layer_depth = 200
#        if simstates[-1,0] > layer_depth and simstates[-1,0] < layer_depth+layer_size:
#            tau=25
#            ku=10

        #Simulation parameters
        deltad=1 #ODE solution resolution
        stepsize=30 #Feet drilled for each step 
        epso=np.arange(0,stepsize,deltad)
        
        #Calculate action as action percentage of input force
        u=(action - (self.numActions - 1) / 2) / (self.numActions - 1) * 2 
        #Set initial values
        [xo,yo,thetao,thetadoto]= simstates[-1,:]
        
        #Calculate theta and thetadot
        alpha=(ku*u+dist)/100;
        exp1=np.exp(-epso/tau);
        theta = (alpha-thetadoto)*tau*exp1 + (epso-tau)*alpha + thetao + tau*thetadoto;
        thetadot = alpha*(-exp1+1)+thetadoto*exp1;
        
        #Calculate x and y
        x = xo + deltad*np.cumsum(np.cos(np.deg2rad(theta)))
        y = yo + deltad*np.cumsum(np.sin(np.deg2rad(theta)))
        
        #Check if bounds have been exceeded
        x_check = ((x<self.stateLow[0]) | (x>self.stateHigh[0]))
        y_check = ((y<self.stateLow[1]) | (y>self.stateHigh[1]))
        trim = (x_check | y_check)
        out_of_bounds = np.any(trim)
        
        #init reward
        reward = 0
        
        #Penalize obstacle collision
#        self.c_x = 520
#        self.c_y = 400
#        self.c_r = 100
#        if np.any((x-self.c_x)**2 + (y - self.c_y)**2 < self.c_r**2):
#            reward=-250
        
        #Truncate states where bounds have been exceeded
        #if all states are out of bounds, keep the first one to avoid empty state arrays
        if trim[0]==True:         
            trim[0]=False
        x = x[~trim].reshape(-1,)
        y = y[~trim].reshape(-1,)
        theta = theta[~trim].reshape(-1,)
        thetadot = thetadot[~trim].reshape(-1,)
        
        simstates = np.vstack((simstates,np.transpose(np.array([x,y,theta,thetadot]))))
        self.state = np.array((x[-1],y[-1],theta[-1],thetadot[-1]*100))
        
        if out_of_bounds:
            self.e_pos = -np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
            self.r_pos = self.e_pos
            self.r_ang = -2*(abs(90-theta[-1]))
            self.r_cur = -np.max(abs(simstates[:,3]*100))
            reward+= self.r_pos + self.r_ang + self.r_cur
            return self.state, reward, True, {}
        else:
            reward+=0
            return self.state, reward, False, {}