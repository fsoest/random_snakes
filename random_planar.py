import networkx as nx
from planar import make_planar_graph
import numpy as np
from random_snakes.snek import random_snake, make_r
from random_snakes.save import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # seed = 42
    n_points = 500
    walk_length = 25
    walks = 1
    # np.random.seed(seed)
    G, points = make_planar_graph(n_points)
    try:
        spl = load_obj('spl_pl_{0}_{1}'.format(n_points, seed))
    except:
        spl = dict(nx.all_pairs_dijkstra_path_length(G))
        # save_obj(spl, 'spl_pl_{0}_{1}'.format(n_points, seed))
    for i in range(walks):
        route, steps = random_snake(G, 0.5, spl, lattice_size=1, points=points, reps=walk_length, print_steps=True)
        r, t = make_r(steps)
        #plt.scatter(t, np.linalg.norm(r, axis=1), marker='.')

        r_abs = points[route[0]] + r
        plt.figure(figsize=(10, 10))
        plt.axis('equal')
        plt.plot(points[:, 0], points[:, 1], 'r.')
        plt.plot(r_abs[:, 0] % 1, r_abs[:, 1] % 1, 'k-')

    plt.xlabel('$t$')
    plt.ylabel('$\langle \ \|\| r(t) \|\|_2 \ \\rangle$ ')
    plt.show()