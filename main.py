
from plothandler import *                           #Import the server that plots the result in browser
from shortest_path_random_walk.shortest_path_random_walk_new import *         #Import everything from shortest_path_random_walk
from stable_baselines import DQN, PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.deepq.policies import MlpPolicy
import numpy as np

def main():
    agent = shortest_path()        #Run the shortest_path program
    
    env = gym.make('well-plot-5-v0')
    env.set_reward(env.scale_point([3,6]),1)
    env.set_init_end_state([1,0],[3,6])
    print("init state: ", env.init_state)
    print("end state: ", env.end_state)
    print("obs space: ", env.observation_space)
    print("shape: ", env.observation_space.shape)
    #model = PPO2('MlpPolicy', env, verbose=1)
    #model.learn(total_timesteps =25000)
    #model.save("ppo2_shortpath")
    
    #test loading
    #del model
    model = PPO2.load("ppo2_shortpath")
    print("done and saved")
    
    #Test trained agent
    policies = []
    obs = env.reset()
    n_steps = 100
    for step in range(n_steps):
        action, _ = model.predict(obs,deterministic=True)
        print("Step {}".format(step+1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=',obs,'reward=',reward,'done=',done)

        #obs = np.clip(obs, 0, (env.grid_width-1)*env.distance_points)
        print("width*dist", env.grid_width*env.distance_points)
        print("obs: ", obs)
        if not env.valid_state(obs):
            print("not valid")
        else:
            print("valid")
        if done:
            print("Goal reached!" , "reward = ", reward)
            break
        policies.append(obs)
        #print("Policies: ", policies)
    print("Policies shape: ",len(policies), "/n policies: ", policies)
    env.plot_path(policies)
    agent.optimal_path(policies)   
    #model.grid_gen()

    #This part starts the plotting server:  
    figure = agent.get_figure()
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
