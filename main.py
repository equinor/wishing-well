from plothandler import *                           #Import the server that plots the result in browser
from shortest_path_random_walk.shortest_path_random_walk_new import *         #Import everything from shortest_path_random_walk
from ppo2_shortpath.ppo2_shortpath import *

def main():
    
    agent = ppo2_shortpath()
    model = agent.get_model()

    path_x, path_y = agent.test_model(model)
    path = np.column_stack([path_x,path_y])
    print("path shape: ",len(path), "\n path shape: ", path.shape, "\n path :\n", path)
    
    #This part starts the plotting server:  
    figure =  agent.env.render(path_x,path_y)
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
