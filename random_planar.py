import networkx as nx
from planar import make_planar_graph, plot_edge
import numpy as np
from random_snakes.snek import random_snake, make_r
from random_snakes.save import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    seed = np.random.randint(0, 255)
    np.random.seed(seed)
    print(seed)


    # seed = 42
    n_points = 100
    walk_length = 10
    walks = 1
    # np.random.seed(seed)
    G, points = make_planar_graph(n_points)
    spl = dict(nx.all_pairs_dijkstra_path_length(G))
    for i in range(walks):
        route, steps = random_snake(G, 0.5, spl, lattice_size=1, points=points, reps=walk_length, verbose=True)
        r, t = make_r(steps)
        #plt.scatter(t, np.linalg.norm(r, axis=1), marker='.')
        r_abs = points[route[0]] + r
        plt.figure(figsize=(10, 10), dpi=300, tight_layout=True)
        plt.axis('equal')

        for edge in G.edges:
            plot_edge(points, edge, c='gray', ls='--')

        plt.plot(points[:, 0], points[:, 1], 'r.')
        plt.plot(r_abs[:, 0], r_abs[:, 1], 'k-')
        plt.scatter(r_abs[0, 0], r_abs[0, 1])

    plt.xlabel('$t$')
    plt.ylabel('$\langle \ \|\| r(t) \|\|_2 \ \\rangle$ ')
    plt.show()
