from plothandler import *                           #Import the server that plots the result in browser
from shortest_path_random_walk.shortest_path_random_walk_new import *         #Import everything from shortest_path_random_walk
from stable_baselines import DQN, PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.deepq.policies import MlpPolicy
import numpy as np
import sys

def main():
    env = gym.make('well-plot-16-v0')
    env.set_reward(env.scale_point([5,6]),1)
    env.set_init_end_state([1,0],[5,6])
    print("init state: ", env.init_state)
    print("end state: ", env.end_state)

    if len(sys.argv)>1:
        #To train model run script with an argument (doesn't matter what)
        model = PPO2('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps =25000)
        model.save("ppo2_shortpath")
    else:
        #Else it will load a saved one
        model = PPO2.load("ppo2_shortpath")
   
    #For plotting
    path_x = []
    path_y = []
    path_x.append(env.init_state[0])
    path_y.append(env.init_state[1])

    #Test trained agent
    obs = env.reset()
    n_steps = 100
    for step in range(n_steps):
        action, _ = model.predict(obs,deterministic=True)
        print("Step {}".format(step+1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=',obs,'reward=',reward,'done=',done)
        if not env.valid_state(obs):
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
        
    path = np.column_stack([path_x,path_y])
    print("path shape: ",len(path), "\n path shape: ", path.shape, "\n path :\n", path)
    
    #This part starts the plotting server:  
    figure =  env.render(path_x,path_y)
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
