
from env.shortest_path_random_walk import *         #Import everything from shortest_path_random_walk


def main():
    model = shortest_path(7, 30, [1, 0], [5, 6])        #Run the shortest_path program
    
    model.grid_gen()
    model.max_pi(1000)  



if __name__ == "__main__":
    main()
