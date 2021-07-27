import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random_snakes.snek import random_snake
from random_snakes.save import load_obj

if __name__ == '__main__':
    np.random.seed(42)
    lattice_size = 50
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)
    # spl = dict(nx.all_pairs_shortest_path_length(square_lattice))
    spl = load_obj('spl_50')
    route, steps = random_snake(square_lattice, 2, spl, lattice_size, reps=100)

    xx, yy = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size))

    # print(make_r(steps))
    # print(np.linalg.norm(make_r(steps), ord=1, axis=1))
    print(square_lattice.edges)
    route = np.array(route)

    plt.scatter(xx, yy, c='k', marker='.', alpha=0.1)
    plt.scatter(route[0, 0], route[0,1], c='b')
    plt.scatter(route[1:, 0], route[1:, 1], c='r')
    plt.show()
