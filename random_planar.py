import networkx as nx
import numpy as np

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r
from random_snakes.save import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    seed = np.random.randint(0, 255)
    np.random.seed(seed)
    print(seed)

    n_points = 100
    t_max = 10
    walks = 10
    d = 0


    planar_graph = PlanarGraph(n_points)
    G = planar_graph.graph
    points = planar_graph.embedding

    spl = dict(nx.all_pairs_dijkstra_path_length(G))
    for i in range(walks):
        route, steps = random_snake(G, d, spl, lattice_size=1, points=points, t_max=10, verbose=True)
        r, t = make_r(steps)

        r_abs = points[route[0]] + r
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300, tight_layout=True)
        ax.axis('equal')

        planar_graph.plot_graph(ax)
        ax.plot(r_abs[:, 0], r_abs[:, 1], 'k-', lw=3)
        ax.scatter(r_abs[0, 0], r_abs[0, 1])

    plt.xlabel('$t$')
    plt.ylabel('$\langle \ \|\| r(t) \|\|_2 \ \\rangle$ ')
    plt.show()
