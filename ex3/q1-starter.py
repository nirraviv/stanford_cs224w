import snap
import numpy as np
import matplotlib.pyplot as plt

def load_graph(name):
    '''
    Helper function to load graphs.
    Use "epinions" for Epinions graph and "email" for Email graph.
    Check that the respective .txt files are in the same folder as this script;
    if not, change the paths below as required.
    '''
    if name == "epinions":
        G = snap.LoadEdgeList(snap.PNGraph, "soc-Epinions1.txt", 0, 1)
    elif name == 'email':
        G = snap.LoadEdgeList(snap.PNGraph, "email-EuAll.txt", 0, 1)   
    else: 
        raise ValueError("Invalid graph: please use 'email' or 'epinions'.")
    return G

def get_in_out_bfs(graph, node, return_count=False):
    bfs_out_tree = snap.GetBfsTree(graph, node, True, False)
    bfs_in_tree = snap.GetBfsTree(graph, node, False, True)
    if return_count: return bfs_out_tree.GetNodes(), bfs_in_tree.GetNodes()
    return bfs_out_tree, bfs_in_tree

def q1_1():
    '''
    You will have to run the inward and outward BFS trees for the 
    respective nodes and reason about whether they are in SCC, IN or OUT.
    You may find the SNAP function GetBfsTree() to be useful here.
    '''
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 2018, compare sizes 
    #and comment on where node 2018 lies.
    G = load_graph("email")
    #Your code here:
    node_id = 2018
    bfs_out_tree, bfs_in_tree = get_in_out_bfs(G, node_id)
    print(f'node {node_id} in email graph:\n{bfs_out_tree.GetNodes()} out nodes\n{bfs_in_tree.GetNodes()} in nodes')
    print(f'The single node in out degree is {next(bfs_in_tree.Nodes()).GetId()}')
    
    ##########################################################################
    
    ##########################################################################
    #TODO: Run outward and inward BFS trees from node 224, compare sizes 
    #and comment on where node 224 lies.
    G = load_graph("epinions")
    #Your code here:
    node_id = 224
    bfs_out_tree, bfs_in_tree = get_in_out_bfs(G, node_id)
    print(f'node {node_id} in epinions graph:\n{bfs_out_tree.GetNodes()} out nodes\n{bfs_in_tree.GetNodes()} in nodes')
    out_nodes = set(n.GetId() for n in bfs_out_tree.Nodes())
    in_nodes = set(n.GetId() for n in bfs_in_tree.Nodes())
    print(f'Number of nodes in the intersection of in-out sets: {len(in_nodes.intersection(out_nodes))}')


    ##########################################################################

    print('1.1: Done!\n')


def plot_reachability(reach_nodes, N, ax=None):
    if ax is None: _, ax = plt.subplots(1,1)
    y = np.logspace(0, np.log10(reach_nodes.max()), N+1)
    y = np.ceil(y)
    x = np.array([np.sum(reach_nodes <= y_) for y_ in y])
    ax.plot(x/N, y)
    ax.set_yscale('log')
    ax.set_xlabel('frac. of starting nodes')
    ax.set_ylabel('number of nodes reached')
    ax.set_xlim(0, 1)

def q1_2():
    '''
    For each graph, get 100 random nodes and find the number of nodes in their
    inward and outward BFS trees starting from each node. Plot the cumulative
    number of nodes reached in the BFS runs, similar to the graph shown in 
    Broder et al. (see Figure in handout). You will need to have 4 figures,
    one each for the inward and outward BFS for each of email and epinions.
    
    Note: You may find the SNAP function GetRndNId() useful to get random
    node IDs (for initializing BFS).
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    N = 100
    scc_email = 0
    scc_epinions = 0
    G_email = load_graph("email")
    G_epinions = load_graph("epinions")
    email_in, email_out, epinions_in, epinions_out = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    number_of_nodes_email = G_email.GetNodes()
    number_of_nodes_epinions = G_epinions.GetNodes()
    for i in range(N):
        # get random node
        email_ind = np.random.randint(number_of_nodes_email)
        epinions_ind = np.random.randint(number_of_nodes_epinions)

        # calculate in and out
        email_out[i], email_in[i] = get_in_out_bfs(G_email, email_ind, True)
        epinions_out[i], epinions_in[i] = get_in_out_bfs(G_epinions, epinions_ind, True)
        if email_out[i] > 100 and email_in[i]>100:
            scc_email += 1
        if epinions_out[i] > 100 and epinions_in[i] > 100:
            scc_epinions += 1
    email_in, email_out = np.sort(email_in), np.sort(email_out)
    epinions_in, epinions_out = np.sort(epinions_in), np.sort(epinions_out)

    # get disconnected
    email_max_wcc = snap.GetMxWcc(G_email)
    email_max_scc = snap.GetMxScc(G_email)
    epinions_max_wcc = snap.GetMxWcc(G_epinions)
    epinions_max_scc = snap.GetMxScc(G_epinions)

    print(f'Email graph\n disconnected: {(number_of_nodes_email - email_max_wcc.GetNodes())/number_of_nodes_email:.2f} - SCC: {email_max_scc.GetNodes()/number_of_nodes_email:.2%}')
    print(f'Epinions graph\n disconnected: {(number_of_nodes_epinions - epinions_max_wcc.GetNodes())/number_of_nodes_epinions:.2f} - SCC: {epinions_max_scc.GetNodes()/number_of_nodes_epinions:.2%}')
    f, ax = plt.subplots(2, 2)

    plot_reachability(email_in, N, ax[0,0])
    ax[0,0].set_title('Email in-links')

    plot_reachability(email_out, N, ax[0,1])
    ax[0,1].set_title('Email out-links')

    plot_reachability(epinions_in, N, ax[1,0])
    ax[1,0].set_title('Epinions in-links')

    plot_reachability(epinions_out, N, ax[1,1])
    ax[1,1].set_title('Epinions out-links')




    ##########################################################################
    print('1.2: Done!\n')


def q1_3():
    '''
    For each graph, determine the size of the following regions:
        DISCONNECTED
        IN
        OUT
        SCC
        TENDRILS + TUBES
        
    You can use SNAP functions GetMxWcc() and GetMxScc() to get the sizes of 
    the largest WCC and SCC on each graph. 
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################
    print('1.3: Done!\n')


def q1_4():
    '''
    For each graph, calculate the probability that a path exists between
    two nodes chosen uniformly from the overall graph.
    You can do this by choosing a large number of pairs of random nodes
    and calculating the fraction of these pairs which are connected.
    The following SNAP functions may be of help: GetRndNId(), GetShortPath()
    '''
    ##########################################################################
    #TODO: See above.
    #Your code here:
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################################
    print('1.4: Done!\n')


if __name__ == "__main__":
    q1_1()
    q1_2()
    q1_3()
    q1_4()
    print("Done with Question 1!\n")
    plt.show()
