import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from random_snakes.graph_generators import LatticeGraph
from random_snakes.snek import make_r
from random_snakes.snek import random_snake


if __name__ == '__main__':
    np.random.seed(42)
    ds = [1, 2, 4, 8, 16]
    walks = 100
    walk_length = 100
    md = np.zeros((len(ds), walks, walk_length))

    # Initialise lattice
    lattice_size = 50
    square_lattice = LatticeGraph(lattice_size).graph

    # Generate the shortest path dict
    spl = dict(nx.all_pairs_shortest_path_length(square_lattice))

    for i, d in enumerate(ds):
        for j in range(walks):
            route, steps = random_snake(square_lattice, d, spl, lattice_size, t_max=walk_length - 1)
            md[i, j] = np.linalg.norm(make_r(steps)[0], ord=1, axis=1)
        plt.plot(np.mean(md[i], axis=0), label=d)
    plt.plot([0, 100], [1, 101], 'k--')
    plt.legend(title='$d$')
    plt.xlabel('$t$')
    plt.ylabel('$\langle \ \|\| r(t) \|\|_1 \ \\rangle$ ')
    plt.show()
