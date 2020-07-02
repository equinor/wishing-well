from stable_baselines import DQN, PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.deepq.policies import MlpPolicy
import numpy as np
import sys
import gym
    
from gym_ww import envs

class ppo2_shortpath:
    def __init__(self):
        self.env = gym.make('well-plot-16-v0')
        self.env.set_reward(self.env.scale_point([5,6]),1)
        self.env.set_init_end_state([1,0],[5,6])
        print("init state: ", self.env.init_state)
        print("end state: ", self.env.end_state)

    def get_model(self):
        if len(sys.argv)>1:
            #To train model run script with an argument (doesn't matter what)
            model = PPO2('MlpPolicy', self.env, verbose=1)
            model.learn(total_timesteps =25000)
            model.save("ppo2_shortpath")
            return model
        else:
            #Else it will load a saved one
            model = PPO2.load("ppo2_shortpath/ppo2_shortpath")
            return model

    def test_model(self,model):
        #For plotting
        path_x = []
        path_y = []
        path_x.append(self.env.init_state[0])
        path_y.append(self.env.init_state[1])

        #Test trained agent
        obs = self.env.reset()
        n_steps = 100
        for step in range(n_steps):
            action, _ = model.predict(obs,deterministic=True)
            print("Step {}".format(step+1))
            print("Action: ", action)
            obs, reward, done, info = self.env.step(action)
            print('obs=',obs,'reward=',reward,'done=',done)
            if not self.env.valid_state(obs):
                print("not valid")
            else:
                print("valid")
            if done:
                print("Goal is reached!" , "reward = ", reward)
                path_x.append(obs[0])
                path_y.append(obs[1])
                break
            path_x.append(obs[0])
            path_y.append(obs[1])    
        return path_x, path_y
        
         

