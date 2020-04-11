###############################################################################
# CS 224W (Fall 2019) - HW3
# Starter code for Problem 3
###############################################################################

import snap
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Setup
num_voters = 10000
decision_period = 10


def read_graphs(path1, path2):
    """
    :param - path1: path to edge list file for graph 1
    :param - path2: path to edge list file for graph 2

    return type: snap.PUNGraph, snap.PUNGraph
    return: Graph 1, Graph 2
    """
    ###########################################################################
    Graph1 = snap.LoadEdgeList(snap.PUNGraph, path1, 0, 1)
    Graph2 = snap.LoadEdgeList(snap.PUNGraph, path2, 0, 1)
    ###########################################################################
    return Graph1, Graph2


def initial_voting_state(Graph):
    """
    Function to initialize the voting preferences.

    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: Python dictionary
    return: Dictionary mapping node IDs to initial voter preference
            ('A', 'B', or 'U')

    Note: 'U' denotes undecided voting preference.

    Example: Some random key-value pairs of the dict are
             {0 : 'A', 24 : 'B', 118 : 'U'}.
    """
    voter_prefs = {}
    ###########################################################################
    for nid in range(num_voters):
        if nid % 10 in range(4):
            voter_prefs[nid] = 'A'
        elif nid % 10 in range(4, 8):
            voter_prefs[nid] = 'B'
        else:
            voter_prefs[nid] = 'U'
    ###########################################################################
    assert(len(voter_prefs) == num_voters)
    return voter_prefs


def iterate_voting(Graph, init_conf):
    """
    Function to perform the 10-day decision process.

    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - init_conf: Dictionary object containing the initial voting
                        preferences (before any iteration of the decision
                        process)

    return type: Python dictionary
    return: Dictionary containing the voting preferences (mapping node IDs to
            'A','B' or 'U') after the decision process.

    Hint: Use global variables num_voters and decision_period to iterate.
    """
    curr_conf = init_conf.copy()
    curr_alternating_vote = 'A'
    ###########################################################################
    for _ in range(10):
        for nid in range(num_voters):
            count = 0
            if init_conf[nid] != 'U': continue
            # get neighbors
            neighbors = snap.TIntV()
            snap.GetNodesAtHop(Graph, nid, 1, neighbors, False)
            # count 'A' vs 'B'
            for n in list(neighbors):
                if curr_conf[n] == 'A': count += 1
                elif curr_conf[n] == 'B': count -= 1
            # decide for current node
            if count > 0: curr_conf[nid] = 'A'
            elif count < 0: curr_conf[nid] = 'B'
            else: curr_conf[nid] = curr_alternating_vote if curr_conf[nid] != curr_alternating_vote else 'B'

        # curr_alternating_vote = 'A' if curr_alternating_vote == 'B' else 'B'

    ###########################################################################
    return curr_conf


def sim_election(Graph):
    """
    Function to simulate the election process, takes the Graph as input and
    gives the final voting preferences (dictionary) as output.
    """
    init_conf = initial_voting_state(Graph)
    conf = iterate_voting(Graph, init_conf)
    return conf


def winner(conf):
    """
    Function to get the winner of election process.
    :param - conf: Dictionary object mapping node ids to the voting preferences

    return type: char, int
    return: Return candidate ('A','B') followed by the number of votes by which
            the candidate wins.
            If there is a tie, return 'U', 0
    """
    ###########################################################################
    count_a = Counter(conf.values())['A']
    count_b = Counter(conf.values())['B']
    if count_a > count_b: return 'A', count_a - count_b
    elif count_a < count_b: return 'B', count_b - count_a
    return 'U', 0
    ###########################################################################


def Q1():
    print ("\nQ1:")
    Gs = read_graphs('graph1.txt', 'graph2.txt')    # List of graphs

    # Simulate election process for both graphs to get final voting preference
    final_confs = [sim_election(G) for G in Gs]

    # Get the winner of the election, and the difference in votes for both
    # graphs
    res = [winner(conf) for conf in final_confs]

    for i in range(2):
        print("In graph %d, candidate %s wins by %d votes" % (i+1, res[i][0], res[i][1]))


def Q2sim(Graph, k):
    """
    Function to simulate the effect of advertising.
    :param - Graph: snap.PUNGraph object representing an undirected graph
             k: amount to be spent on advertising

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    init_conf = initial_voting_state(Graph)
    # change to 'A' according to k
    for nid in range(3000, 3000 + k//100):
        init_conf[nid] = 'A'
    conf = iterate_voting(Graph, init_conf)
    return Counter(conf.values())['A'] - Counter(conf.values())['B']
    ###########################################################################


def find_min_k(diffs):
    """
    Function to return the minimum amount needed for A to win
    :param - diff: list of (k, diff), where diff is the value by which A wins
                   (or loses) i.e. (A-B), for that k.

    return type: int
    return: The minimum amount needed for A to win
    """
    ###########################################################################
    for k, diff in diffs:
        if diff > 0:
            return k
    return None
    ###########################################################################


def makePlot(res, title):
    """
    Function to plot the amount spent and the number of votes the candidate
    wins by
    :param - res: The list of 2 sublists for 2 graphs. Each sublist is a list
                  of (k, diff) pair, where k is the amount spent, and diff is
                  the difference in votes (A-B).
             title: The title of the plot
    """
    Ks = [[k for k, diff in sub] for sub in res]
    res = [[diff for k, diff in sub] for sub in res]
    ###########################################################################
    plt.plot(Ks[0], res[0], label='graph1')
    plt.plot(Ks[1], res[1], label='graph2')
    ###########################################################################
    plt.plot(Ks[0], [0.0] * len(Ks[0]), ':', color='black')
    plt.xlabel('Amount spent ($)')
    plt.ylabel('#votes for A - #votes for B')
    plt.title(title)
    plt.legend()
    plt.show()


def Q2():
    print ("\nQ2:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(1, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q2sim(G, k)) for k in Ks] for G in Gs]

    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in range(2):
        print(formatString.format(i + 1, min_k[i]))

    makePlot(res, 'TV Advertising')


def Q3sim(Graph, k):
    """
    Function to simulate the effect of a dining event.
    :param - Graph: snap.PUNGraph object representing an undirected graph
             k: amount to be spent on the dining event

    return type: int
    return: The number of votes by which A wins (or loses), i.e. (number of
            votes of A - number of votes of B)

    Hint: Feel free to use initial_voting_state and iterate_voting functions.
    """
    ###########################################################################
    init_conf = initial_voting_state(Graph)
    # change to 'A' according to k - high rollers
    result_degree = snap.TIntV()
    snap.GetDegSeqV(Graph, result_degree)
    for nid in np.flip(np.argsort(list(result_degree)))[:k//1000]:
        init_conf[nid] = 'A'
    conf = iterate_voting(Graph, init_conf)
    return Counter(conf.values())['A'] - Counter(conf.values())['B']
    ###########################################################################


def Q3():
    print ("\nQ3:")
    # List of graphs
    Gs = read_graphs('graph1.txt', 'graph2.txt')

    # List of amount of $ spent
    Ks = [x * 1000 for x in range(1, 10)]

    # List of (List of diff in votes (A-B)) for both graphs
    res = [[(k, Q3sim(G, k)) for k in Ks] for G in Gs]

    # List of minimum amount needed for both graphs
    min_k = [find_min_k(diff) for diff in res]

    formatString = "On graph {}, the minimum amount you can spend to win is {}"
    for i in range(2):
        print(formatString.format(i + 1, min_k[i]))

    makePlot(res, 'Wining and Dining')


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    result_degree = snap.TIntV()
    snap.GetDegSeqV(Graph, result_degree)
    result_degree = np.array(result_degree)
    X, Y = np.unique(result_degree, return_counts=True)

    ############################################################################
    return X, Y


def Q4():
    """
    Function to plot the distributions of two given graphs on a log-log scale.
    """
    print ("\nQ4:")
    ###########################################################################
    # List of graphs

    Gs = read_graphs('graph1.txt', 'graph2.txt')
    data = [getDataPointsToPlot(graph) for graph in Gs]

    plt.loglog(data[0][0], data[0][1], linestyle='dotted', color='b', label='graph1')
    plt.loglog(data[1][0], data[1][1], linestyle='dotted', color='r', label='graph2')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of given graphs')
    plt.legend()
    plt.show()

    ###########################################################################


def main():
    Q1()
    Q2()
    Q3()
    Q4()


if __name__ == "__main__":
    main()
