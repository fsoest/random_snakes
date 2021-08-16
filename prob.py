import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r


def betweenness_centrality(g, d):
    sp = dict(nx.all_pairs_dijkstra(g))
    n_nodes = len(g)
    bc = np.zeros(n_nodes)
    start_end = []
    for i in range(n_nodes):
        lengths = sp[i][0]
        paths = sp[i][1]
        for key, val in lengths.items():
            if val != 0 and val <= d:
                path = paths[key]
                start_end.append((path[0], path[-1]))
                if len(path) > 2:
                    for j in np.arange(1, len(path) - 1):
                        bc[path[j]] += 1
    print('Total length of start_end:', len(start_end))
    print('Unique:', len(np.unique(np.array(start_end), axis=0)))
    return  2 * bc / (n_nodes - 1) / (n_nodes - 2)


def linear_fit(coeff, bc, dc, ):
    return coeff[0] * bc + coeff[1] * dc


def main():
    # Seed
    seed = 42
    np.random.seed(seed)
    n_points = 100

    realisations = 100
    t_max = 500

    planar_graph = PlanarGraph(n_points)
    G = planar_graph.graph
    points = planar_graph.embedding
    spl = dict(nx.all_pairs_dijkstra_path_length(G))

    # sp = dict(nx.all_pairs_dijkstra(G))
    # print(sp[0][1][72])

    k = 0
    for i in range(n_points):
        for j in range(n_points):
            if spl[i][j] > 0.5:
                k += 1
    print(k / n_points ** 2)

    ds = [0, 0.1]

    hists = np.zeros((realisations, len(ds),  n_points))
    bins = np.arange(0, n_points + 0.5) - 0.5

    for i in range(realisations):
        for j, d in enumerate(ds):
            route, steps = random_snake(G, d, spl, 1, t_max, points)
            route = np.array(route)
            hists[i, j] = np.histogram(route, bins=bins, density=True)[0]


    mean_hist = np.mean(hists, axis=0)
    print(mean_hist.shape)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, tight_layout=True)

    # for j, d in enumerate(ds):
    #     ax.bar(np.array(bins[1:]) + j * 1/len(ds), mean_hist[j], label=d, width=0.8/len(ds))
    # plt.legend(title='$d$')
    # plt.hlines(np.median(mean_hist), 0, 100, colors=['k'], linestyles='--')

    bc = betweenness_centrality(G, 0.0)
    dc = nx.degree_centrality(G)

    dc_array = np.zeros(n_points)

    for i in range(n_points):
        dc_array[i] = dc[i]

    dc_array /= np.sum(dc_array)

    ax.scatter(mean_hist[1], betweenness_centrality(G, 0.1), marker='.')
    ax.scatter(mean_hist[1], dc_array, marker='.')


    # for d in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     ax.scatter(np.arange(100), betweenness_centrality(G, d), label=d, marker='.')
    # plt.legend()

    # ax.plot(np.arange(n_points) + 0.5, bc_array, color='red', marker='.', linestyle='None')
    # ax.plot(np.arange(n_points) + 0.5, dc_array, color='green', marker='.', linestyle='None')
    plt.show()


if __name__ == '__main__':
    main()