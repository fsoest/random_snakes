from typing import Any
from typing import Tuple

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import networkx as nx
from random_snakes.snek import diff


def plot_edge(node_embedding: np.ndarray, edge: Tuple[Any, Any], **plot_kwargs):
    nodes = node_embedding[edge, :]
    plt.plot(nodes[:, 0], nodes[:, 1], **plot_kwargs)


def make_planar_graph(n_points):
    a = [0, 1, -1]
    points = np.random.uniform(0, 1, (n_points, 2))
    periodic = np.zeros((9 * n_points, 2))

    k = 0
    for i in a:
        for j in a:
            shift = np.zeros((n_points, 2))
            shift[:, 0] = points[:, 0] + i
            shift[:, 1] = points[:, 1] + j
            periodic[k * n_points: (k + 1) * n_points] = shift
            k += 1
    tri = Delaunay(periodic)
    simplices = []
    for simplex in tri.simplices:
        if min(simplex) < n_points:
            simplices.append(simplex)
    simplices = np.array(simplices) % n_points
    simplices = np.sort(simplices, axis=1)
    simplices = np.unique(simplices, axis=0)

    edges = []
    for simplex in simplices:
        edges.append((simplex[0], simplex[1]))
        edges.append((simplex[1], simplex[2]))
        edges.append((simplex[0], simplex[2]))
    edges = np.unique(np.array(edges), axis=0)

    # Create the networkx Graph
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(np.arange(n_points))
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=np.linalg.norm(diff(points[edge[0]], points[edge[1]], 1)))
    return G, points


if __name__ == '__main__':
    # Generate a graph with 10 points
    g, points = make_planar_graph(10)

    # Draw graph with the real node positions
    for edge in g.edges:
        plot_edge(points, edge)
    plt.show()

    # Draw graph with a force layout
    nx.draw(g)
    plt.show()
