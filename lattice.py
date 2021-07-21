import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from snek import random_snake

if __name__ == '__main__':
    np.random.seed(42)
    lattice_size = 50
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)
    spl = dict(nx.all_pairs_shortest_path_length(square_lattice))
    route, steps = random_snake(square_lattice, 8, spl, lattice_size, reps=100)

    xx, yy = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size))

    route = np.array(route)
    print(steps)
    print(len(steps))
    plt.scatter(xx, yy, c='k', marker='.', alpha=0.1)
    plt.scatter(route[0, 0], route[0,1], c='b')
    plt.scatter(route[1:, 0], route[1:, 1], c='r')
    plt.show()
