from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def main():
    np.random.seed(42)
    graph = PlanarGraph(100)
    spl = dict(nx.all_pairs_dijkstra_path_length(graph.graph))
    fig, ax = plt.subplots(nrows=2, ncols=3)#, sharex='all')#, sharey='all')

    ds = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for i, d in enumerate(ds):
        route, steps = random_snake(graph.graph, d, spl, 1, 50, points=graph.embedding)
        r, t = make_r(steps)
        ax.flatten()[i].plot(r[:, 0], r[:, 1], label='$d = {0}$'.format(d))
        ax.flatten()[i].legend()
        ax.flatten()[i].legend()
        ax.flatten()[i].axis('equal')
    plt.tight_layout()
    plt.savefig('figures/planar_route.pdf')
    plt.show()


if __name__ == '__main__':
    main()
