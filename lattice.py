import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from random_snakes.graph_generators import LatticeGraph
from random_snakes.snek import random_snake

if __name__ == '__main__':
    np.random.seed(42)
    lattice_size = 10
    square_lattice = LatticeGraph(lattice_size)

    spl = dict(nx.all_pairs_shortest_path_length(square_lattice.graph))
    route, steps = random_snake(square_lattice.graph, 16, spl, lattice_size, t_max=10)
    route = np.array(route)

    fig, ax = plt.subplots()
    square_lattice.plot_graph(ax)
    ax.scatter(route[0, 0], route[0,1], c='b')
    ax.scatter(route[1:, 0], route[1:, 1], c='r')
    plt.show()
