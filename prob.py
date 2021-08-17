import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r
from functools import partial
from multiprocessing import Pool, cpu_count

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from scipy.optimize import basinhopping


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
    return 2 * bc / (n_nodes - 1) / (n_nodes - 2)


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


def make_data(G, d, spl, t_max, k):
    np.random.seed(k)
    bins = np.arange(0, G.n_points + 0.5) - 0.5
    route, steps = random_snake(G.graph, d, spl, 1, t_max, G.embedding)
    route = np.array(route)
    res = np.zeros((G.n_points, 4))
    res[:, 0] = np.histogram(route, bins=bins, density=True)[0]
    res[:, 1] = G.bc
    res[:, 2] = G.dc
    res[:, 3] = G.bc_inf
    return res


def more_graphs(d):
    reps = 10
    routes = 1000
    t_max = 500
    N = 100

    data = np.zeros((reps, routes, N, 4))

    for i in range(reps):
        np.random.seed(42)
        G = PlanarGraph(N)
        G.bc = betweenness_centrality(G.graph, d)

        dc = nx.degree_centrality(G.graph)
        bc_inf = nx.betweenness_centrality(G.graph, weight='weight')

        G.dc = np.zeros(N)
        G.bc_inf = np.zeros(N)

        for j in range(N):
            G.dc[j] = dc[j]
            G.bc_inf[j] = bc_inf[j]

        spl = dict(nx.all_pairs_dijkstra_path_length(G.graph))
        partial_func = partial(make_data, G, d, spl, t_max)
        # for k in range(routes):
        #     print(partial_func(k).shape)
        #     data[i, k] = partial_func(k)
        with Pool(cpu_count() - 2) as p:
            data[i] = np.array(p.map(partial_func, range(routes)))
    np.save('Data_{0}'.format(d), data.reshape((reps * routes * N, 4)))
    return data.reshape((reps * routes * N, 4))


def plot_data(d):
    try:
        data = np.load('Data_{0}.npy'.format(d))
    except:
        data = more_graphs(d)
        np.save('Data_{0}'.format(d), data)

    p = data[:, 0]
    bc_d = data[:, 1]
    dc = data[:, 2]
    bc_inf = data[:, 3]
    fig, ax = plt.subplots()
    ax.scatter(p, dc, marker='.', label='Degree centrality')
    ax.scatter(p, bc_inf, marker='.', label='Betweenness centrality')
    # ax.scatter(data[:, 3], data[:, 0], marker='.')
    # ax.scatter(bc_inf, bc_d)
    # ax.scatter(bc_inf, dc)
    plt.ylabel('p')
    plt.legend()
    plt.show()


def learn(d):
    data = np.load('Data_{0}.npy'.format(d))
    X = data[:, 1:]
    y = data[:, 0]
    X /= np.max(X)
    y /= np.max(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    svr = SVR()
    svr.fit(X_train, y_train)
    print(svr.score(X_test, y_test))


def cost(p, bc, dc, bc_d, d, x):
    return -1 * np.corrcoef(p, x[0] * dc + x[1] * bc + x[2] * bc_d)[0, 1]


def optimizer(d):
    data = np.load('Data_{0}.npy'.format(d))
    p = data[:, 0]
    dc = data[:, 2]
    bc = data[:, 3]
    bc_d = data[:, 1]
    cost_fun = partial(cost, p, bc, dc, bc_d, d)
    res = basinhopping(cost_fun, [1, 1, 1])
    print(res.fun)
    return res

def calc_correlations(d):
    data = np.load('Data_{0}.npy'.format(d))
    p = data[:, 0]
    bc_d = data[:, 1]
    dc = data[:, 2]
    bc_inf = data[:, 3]
    p_dc = np.corrcoef(p, dc)[0, 1]
    p_bcinf = np.corrcoef(p, bc_inf)[0, 1]
    p_bc_d = np.corrcoef(p, bc_d)[0, 1]
    print('{0}    |  {1}   |  {2} | {3}  '.format(d, np.round(p_dc, 3), np.round(p_bcinf, 3), np.round(p_bc_d, 3)))


if __name__ == '__main__':
    ds = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # for d in ds:
    #     print(d)
    #     more_graphs(d)
    print('d      |      p x dc      |   p x bc_inf   |  p x bc_d')
    for d in ds:
        calc_correlations(d)
        optimizer(d)
    # learn(0.5)