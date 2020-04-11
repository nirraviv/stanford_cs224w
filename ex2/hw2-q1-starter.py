################################################################################
# CS 224W (Fall 2019) - HW1
# Starter code for Question 1
# Last Updated: Sep 25, 2019
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Setup


# Problem 1.1 - Relational Classification
def relational_classification(Graph, num_of_iterations):
    for i in range(num_of_iterations):
        for node in range(1, Graph.number_of_nodes() + 1):
            node = str(node)
            if Graph.nodes[node]['fix']:
                continue
            p = 0.
            for neighbor in nx.neighbors(Graph, node):
                p += Graph.nodes[neighbor]['p']
            p /= len(Graph[node])
            Graph.nodes[node]['p'] = p


def Q1_1(I=2):
    """
    Code for HW2 Q1.1
    """
    G = nx.read_edgelist(r'C:\Users\nirr\Documents\stanford_cs224w\ex2\q1-graph.txt')
    # initialize nodes
    for node in G.nodes:
        if node in ['3', '5']:
            G.nodes[node]['p'] = 1.
            G.nodes[node]['fix'] = True
        elif node in ['8', '10']:
            G.nodes[node]['p'] = 0.
            G.nodes[node]['fix'] = True
        else:
            G.nodes[node]['p'] = 0.5
            G.nodes[node]['fix'] = False
    # run relational classification
    relational_classification(G, I)
    print('(i)')
    for i in ['2','4', '6']:
        print(f'Node {i}: {G.nodes[i]["p"]}')
    print('(ii)')
    for node in G.nodes:
        print(f'Node {node}: {"+" if G.nodes[node]["p"] > 0.5 else "-"}')


# Execute code for Q1.1
Q1_1(I=2)


# Problem 1.2 - Belief Propagation
def get_message(cur_node, nodes, graph, psi, phi, y, prev_node=None):
    m = np.ones((2, 1))
    for n in nodes:
        if prev_node is not None and n == prev_node: continue
        m *= get_message(n, graph[n], graph, psi, phi, y, cur_node)
    if prev_node is None: return m
    cur_psi = psi[cur_node][np.argwhere(np.array(graph[cur_node]) == prev_node).item()]
    if phi[cur_node] is not None:
        cur_phi = phi[cur_node][:, y[cur_node]].reshape((2, 1))
        m *= cur_phi
    m = cur_psi @ m
    return m


def Q1_2():
    """
    Code for Q1.2
    """
    psi_12 = psi_34 = np.array([[1, 0.9], [0.9, 1]])
    psi_23 = psi_35 = np.array([[0.1, 1], [1, 0.1]])
    phi_2 = phi_4 = np.array([[1, .1], [.1, 1]])

    graph = {1: [2], 2: [1, 3], 3: [2, 4, 5], 4: [3], 5: [3]}
    psi = {1: [psi_12], 2: [psi_12.T, psi_23], 3: [psi_23.T, psi_34, psi_35], 4: [psi_34.T], 5: [psi_35.T]}
    phi = {1: None, 2: phi_2, 3: None, 4: phi_4, 5: None}
    y = {2: 0, 4: 1}
    b = {}
    for node, neighbors in graph.items():
        m = get_message(node, neighbors, graph, psi, phi, y)
        if phi[node] is not None:
            m *= phi[node][y[node]].reshape((2,1))
        b[node] = m / m.sum()
    print([f'p(x_{k}=1) = {v[1].item()}' for k,v in b.items()])
    pass


# Execute code for Q1.2
Q1_2()
