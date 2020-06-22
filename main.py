
from plothandler import *                           #Import the server that plots the result in browser
from env.shortest_path_random_walk import *         #Import everything from shortest_path_random_walk

def main():
    model = shortest_path(7, 30, [1, 0], [5, 6])        #Run the shortest_path program
    
    model.grid_gen()
    model.optimal_path(100)   
    model.grid_gen()

    print("heisann")

    #This part starts the plotting server:
    figure = model.get_figure()
    application = MyApplication(figure)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()



if __name__ == "__main__":
    main()
