class Node:
  
    def __init__(self, data, indexloc = None):
        self.data = data
        self.index = indexloc
        
class BinaryTree:

    def __init__(self, nodes = []):
        self.nodes = nodes

    def root(self):
        return self.nodes[0]
    
    def iparent(self, i):
        return (i - 1) // 2
    
    def ileft(self, i):
        return 2*i + 1

    def iright(self, i):
        return 2*i + 2

    def left(self, i):
        return self.node_at_index(self.ileft(i))
    
    def right(self, i):
        return self.node_at_index(self.iright(i))

    def parent(self, i):
        return self.node_at_index(self.iparent(i))

    def node_at_index(self, i):
        return self.nodes[i]

    def size(self):
        return len(self.nodes)

class DijkstraNodeDecorator:
    
    def __init__(self, node):
        self.node = node
        self.prov_dist = float('inf')
        self.hops = []

    def index(self):
        return self.node.index

    def data(self):
        return self.node.data
    
    def update_data(self, data):
        self.prov_dist = data['prov_dist']
        self.hops = data['hops']
        return self

class MinHeap(BinaryTree):

    def __init__(self, nodes, is_less_than = lambda a,b: a < b, get_index = None, update_node = lambda node, newval: newval):
        BinaryTree.__init__(self, nodes)
        self.order_mapping = list(range(len(nodes)))
        self.is_less_than, self.get_index, self.update_node = is_less_than, get_index, update_node
        self.min_heapify()

    # Heapify at a node assuming all subtrees are heapified
    def min_heapify_subtree(self, i):

        size = self.size()
        ileft = self.ileft(i)
        iright = self.iright(i)
        imin = i
        if( ileft < size and self.is_less_than(self.nodes[ileft], self.nodes[imin])):
            imin = ileft
        if( iright < size and self.is_less_than(self.nodes[iright], self.nodes[imin])):
            imin = iright
        if( imin != i):
            self.nodes[i], self.nodes[imin] = self.nodes[imin], self.nodes[i]
            # If there is a lambda to get absolute index of a node
            # update your order_mapping array to indicate where that index lives
            # in the nodes array (so lookup by this index is O(1))
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[imin])] = imin
                self.order_mapping[self.get_index(self.nodes[i])] = i
            self.min_heapify_subtree(imin)


    # Heapify an un-heapified array
    def min_heapify(self):
        for i in range(len(self.nodes), -1, -1):
            self.min_heapify_subtree(i)

    def min(self):
        return self.nodes[0]

    def pop(self):
        min = self.nodes[0]
        if self.size() > 1:
            self.nodes[0] = self.nodes[-1]
            self.nodes.pop()
            # Update order_mapping if applicable
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[0])] = 0
            self.min_heapify_subtree(0)
        elif self.size() == 1: 
            self.nodes.pop()
        else:
            return None
        # If self.get_index exists, update self.order_mapping to indicate
        # the node of this index is no longer in the heap
        if self.get_index is not None:
            # Set value in self.order_mapping to None to indicate it is not in the heap
            self.order_mapping[self.get_index(min)] = None
        return min

    # Update node value, bubble it up as necessary to maintain heap property
    def decrease_key(self, i, val):
        self.nodes[i] = self.update_node(self.nodes[i], val)
        iparent = self.iparent(i)
        while( i != 0 and not self.is_less_than(self.nodes[iparent], self.nodes[i])):
            self.nodes[iparent], self.nodes[i] = self.nodes[i], self.nodes[iparent]
            # If there is a lambda to get absolute index of a node
            # update your order_mapping array to indicate where that index lives
            # in the nodes array (so lookup by this index is O(1))
            if self.get_index is not None:
                self.order_mapping[self.get_index(self.nodes[iparent])] = iparent
                self.order_mapping[self.get_index(self.nodes[i])] = i
            i = iparent
            iparent = self.iparent(i) if i > 0 else None

    def index_of_node_at(self, i):
        return self.get_index(self.nodes[i])

class Graph: 

    def __init__(self, nodes):
        self.adj_list = [ [node, []] for node in nodes ]
        for i in range(len(nodes)):
            nodes[i].index = i


    def connect_dir(self, node1, node2, weight = 1):
        node1, node2 = self.get_index_from_node(node1), self.get_index_from_node(node2)
        # Note that the below doesn't protect from adding a connection twice
        self.adj_list[node1][1].append((node2, weight))

    def connect(self, node1, node2, weight = 1):
        self.connect_dir(node1, node2, weight)
        self.connect_dir(node2, node1, weight)

    
    def connections(self, node):
        node = self.get_index_from_node(node)
        return self.adj_list[node][1]
    
    def get_index_from_node(self, node):
        if not isinstance(node, Node) and not isinstance(node, int):
            raise ValueError("node must be an integer or a Node object")
        if isinstance(node, int):
            return node
        else:
            return node.index

    def dijkstra(self, src):
        
        src_index = self.get_index_from_node(src)
        # Map nodes to DijkstraNodeDecorators
        # This will initialize all provisional distances to infinity
        dnodes = [ DijkstraNodeDecorator(node_edges[0]) for node_edges in self.adj_list ]
        # Set the source node provisional distance to 0 and its hops array to its node
        dnodes[src_index].prov_dist = 0
        dnodes[src_index].hops.append(dnodes[src_index].node)
        # Set up all heap customization methods
        is_less_than = lambda a, b: a.prov_dist < b.prov_dist
        get_index = lambda node: node.index()
        update_node = lambda node, data: node.update_data(data)

        #Instantiate heap to work with DijkstraNodeDecorators as the hep nodes
        heap = MinHeap(dnodes, is_less_than, get_index, update_node)

        min_dist_list = []

        while heap.size() > 0:
            # Get node in heap that has not yet been "seen"
            # that has smallest distance to starting node
            min_decorated_node = heap.pop()
            min_dist = min_decorated_node.prov_dist
            hops = min_decorated_node.hops
            min_dist_list.append([min_dist, hops])
            
            # Get all next hops. This is no longer an O(n^2) operation
            connections = self.connections(min_decorated_node.node)
            # For each connection, update its path and total distance from 
            # starting node if the total distance is less than the current distance
            # in dist array
            for (inode, weight) in connections: 
                node = self.adj_list[inode][0]
                heap_location = heap.order_mapping[inode]
                if(heap_location is not None):
                    tot_dist = weight + min_dist
                    if tot_dist < heap.nodes[heap_location].prov_dist:
                        hops_cpy = list(hops)
                        hops_cpy.append(node)
                        data = {'prov_dist': tot_dist, 'hops': hops_cpy}
                        heap.decrease_key(heap_location, data)

        return min_dist_list 

#Custom import

import numpy as np

#Variables
n_rows = 10
n_columns = 10
n_nodes = n_rows*n_columns
idx_start_node = 1
idx_goal_node = 99

weights = np.ones(n_nodes)
weights[20:40] = 2
weights[40:50] = 3
weights[50:70] = 4
weights[73:80] = 5
weights[84]    = 5
weights[96]    = 5

#Initialize nodes
node_list = []

for i in range(n_nodes):
    node_i = Node("{0:b}".format(i)) #creates nodes with boolean string as varable name (etc. 0 ->'0', 4 -> '100')
    node_list.append(node_i)
    
    
g = Graph(node_list)
for i in range(n_nodes): #Initialize all edges to node neighbours (max 8) given weights
    if (i + 1)%n_columns != 0:              #not in last column
        g.connect(node_list[i],node_list[i+1],1*(weights[i]+ weights[i+1])/2)
        if (i)//n_columns != (n_rows -1):   #...and not in last row 
            g.connect(node_list[i],node_list[i+n_columns+1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns+1])/2)
        if (i)//n_columns != 0:             #...and not in first row
            g.connect(node_list[i],node_list[i-n_columns+1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns+1])/2)
    
    if (i + 1)%n_columns != 1:              #not in first column
        g.connect(node_list[i],node_list[i-1],1*(weights[i]+ weights[i-1])/2)
        if (i)//n_columns != (n_rows -1):   #...and not in last row 
            g.connect(node_list[i],node_list[i+n_columns-1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns-1])/2)
        if (i)//n_columns != 0:             #...and not in first row
            g.connect(node_list[i],node_list[i-n_columns-1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns-1])/2)
    
    if (i)//n_columns != 0:                 #not in first row
        g.connect(node_list[i],node_list[i-n_columns],1*(weights[i]+ weights[i-n_columns])/2)
    
    if (i)//n_columns != (n_rows -1):       #not in last row
        g.connect(node_list[i],node_list[i+n_columns],1*(weights[i]+ weights[i+n_columns])/2)

source = node_list[idx_start_node]
print([(weight, [n.data for n in node]) for (weight, node) in g.dijkstra(source)][idx_goal_node])

def coordinates(idx_str,n_columns, n_rows):
    idx_dec = (int(idx_str, 2))
    y = idx_dec//n_columns
    x = idx_dec - y*n_columns
    return x,y

#Calculate shortest path

traj_str = [(weight, [n.data for n in node]) for (weight, node) in g.dijkstra(source)][idx_goal_node][1]
traj_coord = []
for i in range(len(traj_str)):
    traj_coord.append(coordinates(traj_str[i], n_columns, n_rows))
print("shortest path trajectory:", traj_coord)


#Illustration 

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

colors = 'lime red blue magenta yellow'.split()
cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)

def illustrate(weights,traj_list):
    w_m = np.reshape(weights,(n_rows,n_columns))
    ax = plt.subplot(111)
    im = plt.imshow(w_m, cmap=cmap)
    for i in range(len(traj_list)):
        point = traj_list[i]
        plt.scatter(point[0],point[1])
        if i > 0:
            point_prev = traj_list[i-1]
            plt.plot([point_prev[0],point[0]], [point_prev[1],point[1]])


    divider = make_axes_locatable(ax)       
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label = "Weights")  
    plt.show()
    return

illustrate(weights,traj_coord)