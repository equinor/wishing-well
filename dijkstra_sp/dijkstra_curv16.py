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

#Import
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import timeit

def init_nodes(n_nodes):
    node_list = []
    for i in range(n_nodes):        
        node_i = Node("{0:b}".format(i)) #creates nodes with boolean string as varable name (etc. 0 ->'0', 4 -> '100')
        node_list.append(node_i)  
    graph = Graph(node_list)
    return graph, node_list


def init_edges(graph, node_list, weights, n_columns, n_rows, v_prom, h_prom, bool_16, idx_start_node, idx_goal_node):
    edges = []
    for i in range(len(node_list)): #Initialize all edges to node neighbours given weights
        #==First 8 edges== 
        #Right(3)
        if (i + 1)%n_columns != 0:              #not in last column
            if (i+1,1*(weights[i]+ weights[i+1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist
                graph.connect(node_list[i],node_list[i+1],1*(weights[i]+ weights[i+1])/2)
                edges.append((i,i+1)), edges.append((i+1,i))
            if (i)//n_columns != (n_rows -1):   #...and not in last row 
                if (i+n_columns+1,np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns+1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist
                    graph.connect(node_list[i],node_list[i+n_columns+1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns+1])/2)
                    edges.append((i,i+n_columns+1)), edges.append((i+n_columns+1,i))
            if (i)//n_columns != 0:             #...and not in first row
                if (i+n_columns+1,np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns+1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist
                    graph.connect(node_list[i],node_list[i-n_columns+1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns+1])/2)
                    edges.append((i,i-n_columns+1)), edges.append((i-n_columns+1,i))
        #Left(3)
        if (i + 1)%n_columns != 1: #not in first column
            if (i-1,1*(weights[i]+ weights[i-1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist      
                    graph.connect(node_list[i],node_list[i-1],1*(weights[i]+ weights[i-1])/2)
                    edges.append((i,i-1)), edges.append((i-1,i))
            if (i)//n_columns != (n_rows -1):   #...and not in last row
                if (i+n_columns-1,np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns-1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist 
                    graph.connect(node_list[i],node_list[i+n_columns-1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i+n_columns-1])/2)
                    edges.append((i,i+n_columns-1)), edges.append((i+n_columns-1,i))
            if (i)//n_columns != 0:             #...and not in first row
                if (i-n_columns-1,np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns-1])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist 
                    graph.connect(node_list[i],node_list[i-n_columns-1],np.round(np.sqrt(2),2)*(weights[i]+ weights[i-n_columns-1])/2)
                    edges.append((i,i-n_columns-1)), edges.append((i-n_columns-1,i))
        
        #Up(1)
        if (i)//n_columns != 0: #not in first row
            if (i-n_columns,1*(weights[i]+ weights[i-n_columns])/2) not in graph.connections(node_list[i]): #This edge doesn't already exist                       
                graph.connect(node_list[i],node_list[i-n_columns],1*(weights[i]+ weights[i-n_columns])/2)
                edges.append((i,i-n_columns)), edges.append((i-n_columns,i))
        #Down(1)
        if (i)//n_columns != (n_rows -1):       #not in last row
            if (i+n_columns,1*(weights[i]+ weights[i+n_columns])/2) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                graph.connect(node_list[i],node_list[i+n_columns],1*(weights[i]+ weights[i+n_columns])/2)
                edges.append((i,i+n_columns)), edges.append((i+n_columns,i))
        
        if bool_16 == True: #Extend to 16 edges
            #Left(2)
            if (i)%n_columns > 1:               #not in two first columns
                if (i)//n_columns != (n_rows -1):   #...and not in last row
                    if (i+n_columns-2,np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns-2] + weights[i+n_columns-1] + weights[i-1])/4) not in graph.connections(node_list[i]): #This edge doesn't already exist 
                        graph.connect(node_list[i],node_list[i+n_columns-2],np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns-2] + weights[i+n_columns-1] + weights[i-1])/4)
                        edges.append((i,i+n_columns-2)), edges.append((i+n_columns-2,i))
                if (i)//n_columns != 0:             #...and not in first row
                    if (i-n_columns-2,np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns-2] + weights[i-n_columns-1] + weights[i-1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i-n_columns-2],np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns-2] + weights[i-n_columns-1] + weights[i-1])/4)
                        edges.append((i,i-n_columns-2)), edges.append((i-n_columns-2,i))
            #Right(2)   
            if (i)%n_columns < (n_columns - 2): #not in two last columns
                if (i)//n_columns != (n_rows -1):   #...and not in last row
                    if (i+n_columns+2,np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns+2] + weights[i+n_columns-+1] + weights[i+1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i+n_columns+2],np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns+2] + weights[i+n_columns-+1] + weights[i+1])/4)
                        edges.append((i,i+n_columns+2)), edges.append((i+n_columns+2,i))
                if (i)//n_columns != 0:             #...and not in first row
                    if (i-n_columns+2,np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns+2] + weights[i-n_columns+1] + weights[i+1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i-n_columns+2],np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns+2] + weights[i-n_columns+1] + weights[i+1])/4)
                        edges.append((i,i-n_columns+2)), edges.append((i-n_columns+2,i))
            #Up(2)
            if (i)//n_columns > 1:              #not in first two rows
                if (i + 1)%n_columns != 0:           #...and not in last column
                    if (i-2*n_columns+1,np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns] + weights[i-n_columns+1] + weights[i-2*n_columns+1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i-2*n_columns+1],np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns] + weights[i-n_columns+1] + weights[i-2*n_columns+1])/4)
                        edges.append((i,2*n_columns+1)), edges.append((2*n_columns+1,i))
                if (i + 1)%n_columns != 1:           #...and not in first column
                    if (i-2*n_columns-1,np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns] + weights[i-n_columns+-1] + weights[i-2*n_columns-1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i-2*n_columns-1],np.round(np.sqrt(5),2)*(weights[i]+ weights[i-n_columns] + weights[i-n_columns+-1] + weights[i-2*n_columns-1])/4)
                        edges.append((i,i-2*n_columns-1)), edges.append((i-2*n_columns-1,i))
            #Down(2)
            if (i)//n_columns < (n_rows -2):    #not in last two rows
                if (i + 1)%n_columns != 0:           #...and not in last column
                    if (i+2*n_columns+1,np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns] + weights[i+n_columns+1] + weights[i+2*n_columns+1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i+2*n_columns+1],np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns] + weights[i+n_columns+1] + weights[i+2*n_columns+1])/4)
                        edges.append((i,i+2*n_columns+1)), edges.append((i+2*n_columns+1,i))
                if (i + 1)%n_columns != 1:           #...and not in first column
                    if (i+2*n_columns-1,np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns] + weights[i+n_columns+-1] + weights[i+2*n_columns-1])/4) not in graph.connections(node_list[i]):    #This edge doesn't already exist
                        graph.connect(node_list[i],node_list[i+2*n_columns-1],np.round(np.sqrt(5),2)*(weights[i]+ weights[i+n_columns] + weights[i+n_columns+-1] + weights[i+2*n_columns-1])/4)
                        edges.append((i,i+2*n_columns-1)), edges.append((i+2*n_columns-1,i))
        #Promote vertical start of trajectory
        if i == idx_start_node and (i)//n_columns != (n_rows -1):   #...and not in last row
            graph.connect(node_list[i],node_list[i+n_columns],1*(weights[i]+ weights[i+n_columns])/2-v_prom)
            edges.append((i,i+n_columns)), edges.append((i+n_columns,i))

        #Promote horizontal end of trajectory
        if i == idx_goal_node: 
            if (i + 1)%n_columns != 0:           #...and not in last column
                graph.connect(node_list[i],node_list[i+1],1*(weights[i]+ weights[i+1])/2-h_prom)
                edges.append((i,i+1)), edges.append((i+1,i))

            if (i + 1)%n_columns != 1:           #...and not in first column
                graph.connect(node_list[i],node_list[i-1],1*(weights[i]+ weights[i-1])/2-h_prom)
                edges.append((i,i-1)), edges.append((i-1,i))
    return graph, edges


def str_to_coord(idx_str, n_columns, n_rows):
    idx_dec = (int(idx_str, 2))
    y = idx_dec//n_columns
    x = idx_dec - y*n_columns
    return x,y


def idx_to_coord(idx,n_columns, n_rows):
    y = idx//n_columns
    x = idx - y*n_columns
    return x,y


def tuple_string(tup):
    return "{0:b}".format(tup[0])+"-"+"{0:b}".format(tup[1])


def nodes_to_angle(pair_a, pair_b,n_columns, n_rows):
    a = np.array(idx_to_coord(pair_a[0],n_columns,n_rows))
    b = np.array(idx_to_coord(pair_a[1],n_columns,n_rows))
    if pair_b[0] not in pair_b:
        c = np.array(idx_to_coord(pair_b[0],n_columns,n_rows))
    else: c = np.array(idx_to_coord(pair_b[1],n_columns,n_rows))
    ab = b - a
    bc = c - b
    return np.arccos(np.dot(ab,bc)/(np.linalg.norm(ab)*np.linalg.norm(bc))) 


def init_pair_nodes(graph, node_list, n_nodes, edges):
    n_pair_nodes = len(edges)
    node_pair_list = []                                #List of nodes for pair-graph
    node_pair_dir = {}                                 #Directory to keep track of the corresponding node-pair indices and previous weight for each node-pair-tuple
    for i in range(n_pair_nodes):                      #Create nodes corresponding to each previous edge
        edge_w = np.inf   
        for connection in graph.connections(node_list[edges[i][0]]):
            if (connection[0] == edges[i][1]) and (connection[1]<edge_w):
                edge_w = connection[1]

        node_pair_dir[tuple_string(edges[i])] = (i,edge_w)
        node_pair_i = Node(tuple_string(edges[i])) #creates nodes from previous edges with boolean string as varable name (etc. 2-4 -> '10-100')
        node_pair_list.append(node_pair_i)
    
    node_pair_list.append(Node("start_node"))
    node_pair_list.append(Node("target_node"))
    g_pairs = Graph(node_pair_list)
 
    return g_pairs, node_pair_list, node_pair_dir


def init_pair_edges(g_pairs, graph, node_pair_list, node_pair_dir, edges, idx_start_node, idx_goal_node, curv_pen, n_col, n_rows):
    for i in range(len(edges)):    #Initialize all edges to node-pair neighbours given previous individual weights and angle between node-pair-lines
        node = edges[i][1]   #Connecting node                      
        if idx_start_node == edges[i][0]:
            g_pairs.connect(node_pair_list[i],node_pair_list[-2],1)
        if idx_goal_node == edges[i][1]:
            g_pairs.connect(node_pair_list[i],node_pair_list[-1],1)
        
        for connection in graph.connections(node):
            if connection[0] in edges[i]:
                continue
            edge_weight = node_pair_dir[tuple_string(edges[i])][1] + node_pair_dir[tuple_string((node,connection[0]))][1] + curv_pen*nodes_to_angle(edges[i], (node,connection[0]), n_col,n_rows)**2
            if (node_pair_dir[tuple_string((node,connection[0]))][0], edge_weight) not in g_pairs.connections(node_pair_list[i]):         #This edge doesn't already exist
                g_pairs.connect_dir(node_pair_list[i], node_pair_list[node_pair_dir[tuple_string((node,connection[0]))][0]], edge_weight)

    return g_pairs


def dijkstraSPA(graph_pairs, node_pair_list): #Calculate shortest path
    source = node_pair_list[-2]
    paths = [(weight, [n.data for n in node]) for (weight, node) in graph_pairs.dijkstra(source)] #All paths from start node
    for i in range(len(node_pair_list)): #Find the path from start node to goal node 
        if (paths[i][1][-1] == "target_node"):
            traj_str = paths[i][1]
            traj_cost = paths[i][0]
            return traj_str, traj_cost
    print("Error: Unable to find sjortest path")
    return

def illustrate(weights,traj_str,n_columns,n_rows):
    colors = 'lime red cyan magenta yellow blue'.split()
    cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)
    w_m = np.reshape(weights,(n_rows,n_columns))
    ax = plt.subplot(111)
    im = plt.imshow(w_m, cmap=cmap)
    for i in range(1,len(traj_str)-1):
        line = traj_str[i].split('-')
        p1 = str_to_coord(line[0],n_columns,n_rows)
        p2 = str_to_coord(line[1],n_columns,n_rows)
        plt.plot([p1[0], p2[0]], [p1[1],p2[1]])

    divider = make_axes_locatable(ax)       
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label = "Weights")  
    plt.show()
    return

def run_DSPA(n_rows, n_col, idx_start, idx_target, weights, v_prom, h_prom, curv_pen , bool_16):
    start = timeit.default_timer()

    n_nodes = n_rows*n_col
    graph, nodes = init_nodes(n_nodes)
    graph, edges = init_edges(graph, nodes, weights, n_col, n_rows, v_prom, h_prom, bool_16,idx_start,idx_target)
    graph_pairs, node_pair_list, node_pair_dir = init_pair_nodes(graph, nodes, n_nodes, edges)
    graph_pairs = init_pair_edges(graph_pairs, graph, node_pair_list, node_pair_dir, edges ,idx_start,idx_target,curv_pen, n_col, n_rows)
    traj, cost = dijkstraSPA(graph_pairs, node_pair_list)

    stop = timeit.default_timer()
    print("Time: ", stop-start) 

    illustrate(weights,traj,n_col,n_rows)
    return traj, cost


#=======SET PARAMETERS==========# 

n_rows = 13
n_col = 10
idx_start = 1
idx_target = 99

weights = np.ones(n_rows*n_col)
weights[20:40] = 2
weights[40:50] = 3
weights[50:70] = 4
weights[73:80] = 5
weights[84]    = 5
weights[96]    = 5
weights[100:130] = 6

v_start_prom = 2
h_end_prom = 2
curv_pen = 3
bool_16 = True # Connectivity of 16 (connectivity of 8 if false)

traj, cost = run_DSPA(n_rows, n_col, idx_start, idx_target, weights, v_start_prom, h_end_prom, curv_pen , bool_16)
