from random_snakes.graph_generators import LatticeGraph
from random_snakes.snek import random_snake
import networkx as nx
import matplotlib.pyplot as plt
from random_snakes.save import *
import numpy as np


def main():
    np.random.seed(42)
    graph = LatticeGraph(50)
    try:
        spl = load_obj('spl')
    except:
        spl = dict(nx.all_pairs_shortest_path_length(graph.graph))
        save_obj(spl, 'spl')

    r2, d2 = random_snake(graph.graph, 2, spl, graph.lattice_size, t_max=50)
    r8, d8 = random_snake(graph.graph, 8, spl, graph.lattice_size, t_max=50)
    r16, d16 = random_snake(graph.graph, 16, spl, graph.lattice_size, t_max=50)

    fig, ax = plt.subplots()
    graph.plot_graph(ax)
    r2 = np.array(r2)
    r8 = np.array(r8)
    r16 = np.array(r16)
    plt.scatter(r2[:, 0], r2[:, 1], marker='.', label='2')
    plt.scatter(r8[:, 0], r8[:, 1], marker='.', label='8')
    plt.scatter(r16[:, 0], r16[:, 1], marker='.', label='16')
    plt.legend(title='$d$')
    plt.axis('off')
    plt.savefig('figures/grid_route.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()