
from plothandler import *                           #Import the server that plots the result in browser
from shortest_path_random_walk.shortest_path_random_walk_new import *         #Import everything from shortest_path_random_walk

def main():
    #agent = shortest_path()        #Run the shortest_path program
    
    #agent.optimal_path()   

    env = gym.make('well-plot-21-v0')
    #env.set_grid_size(100,50,200)
    #env.set_init_end_state((10,0),(80,45))
    #check_env(agent.env,warn=True)

    env.reset()
    state_list = [env.state]
    #action = 1.0
    #new_state, reward, done, info = env.step(action)
    #state_list.append(new_state)
    for _ in range(100):
        action = -1.0
        new_state, reward, done, info = env.step(action)
        state_list.append(new_state)

    env.plot_path(state_list)

    figure = env.render()

    #This part starts the plotting server:
    #figure = agent.get_figure()
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
