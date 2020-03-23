################################################################################
# CS 224W (Fall 2019) - HW1
# Starter code for Question 2
# Last Updated: Sep 25, 2019
################################################################################

import snap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks

# setup
science_net = None
sim_mat = None
neighbors = None
V_tilde = None


# Problem 2.1
def get_basic_feaures(Graph):
    """
    :param - Graph: snap.TUNGraph

    return: each node degree, ego edges and subnet edges and neighbors dictionary
    """
    ############################################################################
    # TODO: Your code here!
    def get_subgraph_edges(G, nodes):
        NIdV = snap.TIntV()
        for n in nodes:
            NIdV.Add(n)
        SubGraph = snap.GetSubGraph(G, NIdV)
        return SubGraph.GetEdges()

    features = np.zeros((Graph.GetNodes(), 4))
    neighbors_dict = {}
    for i, node in enumerate(Graph.Nodes()):
        node_id = node.GetId()
        features[i, 0] = node_id
        # get 1st and 2nd neighbors
        first_n = snap.TIntV()
        second_n = snap.TIntV()
        snap.GetNodesAtHop(Graph, node_id, 1, first_n, False)
        snap.GetNodesAtHop(Graph, node_id, 2, second_n, False)
        first_n = list(first_n)
        second_n = list(second_n)
        neighbors_dict[node_id] = first_n
        # get node degree
        features[i, 1] = len(first_n)
        # get node num of edges in the node egonet
        features[i, 2] = get_subgraph_edges(Graph, [node_id] + first_n)
        # get edges out of egonet
        total_edges = get_subgraph_edges(Graph, [node_id] + first_n + second_n)
        features[i, 3] = total_edges - features[i, 2]

    ############################################################################
    return features, neighbors_dict


def get_similarity(features):
    """
    :param - features: features matrix of all graph nodes [#nodes, #features]

    return: similarity matrix
    """
    ############################################################################
    # TODO: Your code here!
    sim_mat = cosine_similarity(features, features)
    ############################################################################
    return sim_mat


def Q2_1(node):
    """
    Code for HW1 Q2.1
    """
    global science_net, neighbors, V_tilde

    science_net = snap.TUNGraph.Load(snap.TFIn("hw1-q2.graph"))
    # calculate basic features
    V_tilde, neighbors = get_basic_feaures(science_net)
    sim_mat = get_similarity(V_tilde[:, 1:])

    node_row = np.argwhere(V_tilde[:, 0] == node).item()
    v9 = sim_mat[node_row]
    v9[node_row] = 0
    print(f'node {node} - \nfeature vector: {V_tilde[node_row, 1:].tolist()}')
    print(f'top-5 similar nodes: {np.argsort(v9)[-5:].tolist()}')


# Execute code for Q2.1
Q2_1(9)


def get_recursive_features(features, n_dict, K=1):
    for _ in range(K):
        mean_features = np.zeros_like(features[:, 1:])
        sum_features = np.zeros_like(features[:, 1:])
        tmp = features.copy()
        for node in features[:, 0]:
            node = int(node)
            # get neighbors
            neighbor = n_dict[node]
            # get mean and sum of neighbors
            mean_features[node] = features[neighbor, 1:].mean(0) if len(neighbor) else np.zeros(mean_features.shape[1])
            sum_features[node] = features[neighbor, 1:].sum(0) if len(neighbor) else np.zeros(sum_features.shape[1])
        # concatenate
        features = np.concatenate([tmp, mean_features, sum_features], axis=1)
    return features


# Problem 2.2 - Clustering Coefficient
def Q2_2(K, node):
    """
    Code for Q2.2
    """
    global V_tilde, neighbors, sim_mat
    V_tilde = get_recursive_features(V_tilde, neighbors, K)
    sim_mat = get_similarity(V_tilde[:, 1:])
    node_row = np.argwhere(V_tilde[:, 0] == node).item()
    v9 = sim_mat[node_row]
    v9[node_row] = 0
    print(f'node {node} - \nfeature vector: {V_tilde[node_row, 1:].tolist()}')
    print(f'neighbors: {neighbors[node]}')
    print(f'top-5 similar nodes: {np.argsort(v9)[-5:].tolist()}')


# Execute code for Q2.2
Q2_2(2, 9)


def draw_roles_subgraph(Graph, V, node):
    def get_subgraph(G, n):
        first_n = snap.TIntV()
        second_n = snap.TIntV()
        snap.GetNodesAtHop(Graph, n, 1, first_n, False)
        snap.GetNodesAtHop(Graph, n, 2, second_n, False)
        draw_nodes = [n] + list(first_n) + list(second_n)
        NIdV = snap.TIntV()
        for i in draw_nodes:
            NIdV.Add(i)
        SubGraph = snap.GetSubGraph(Graph, NIdV)
        return SubGraph

    # draw given node subgraph
    sub_graph = get_subgraph(Graph, node)
    snap.DrawGViz(sub_graph, snap.gvlDot, f'{node}_subgraph.png', f'{node} subgraph', True)

    hist, bin_edges = np.histogram(V, bins=20)
    # get peaks
    peaks = find_peaks(hist, threshold=10)[0].tolist()
    if hist[0] > 10:
        peaks += [0]
    peaks += [19]
    # get and draw subgraphs
    for peak in peaks:
        ind = np.argwhere((V<bin_edges[peak+1]) & (V>=bin_edges[peak])).flatten()
        draw_node = int(np.random.choice(ind))
        sub_graph = get_subgraph(Graph, draw_node)
        snap.DrawGViz(sub_graph,snap.gvlNeato, f'{draw_node}_subgraph.png', f'{draw_node} subgraph with sim={V[draw_node]}', True)


# Problem 2.3 - Clustering Coefficient
def Q2_3(node):
    """
    Code for Q2.3
    """
    global sim_mat, science_net
    node_row = np.argwhere(V_tilde[:, 0] == node).item()
    v = sim_mat[node_row]
    # draw given node cosine similarity histogram
    plt.hist(v, bins=20)
    plt.xlabel(f'Cosine similarity')
    plt.ylabel('Counts')
    plt.title(f'Node {node} similarity histogram')

    draw_roles_subgraph(science_net, v, node)

    plt.show()


# Execute code for Q2.3
Q2_3(9)
