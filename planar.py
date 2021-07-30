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


def make_planar_graph(n_points: int, lattice_size: float = 1.):
    # Randomly draw points from the defined bounds
    points = np.random.uniform(0, lattice_size, (n_points, 2))

    # Create empty array to store the periodic extension
    periodic = np.zeros((9 * n_points, 2))

    # Extend the lattice in a 3x3 array
    a = [0, 1, -1]
    k = 0
    for i in a:
        for j in a:
            shift = points + np.array([i, j]) * lattice_size
            periodic[k * n_points: (k + 1) * n_points] = shift
            k += 1

    # Calculate the Delaunay triangulation over the periodic extension
    tri = Delaunay(periodic)

    # Determine the triangles that contain at least one node in the original lattice
    simplices = [
        simplex for simplex in tri.simplices
        if min(simplex) < n_points
    ]

    # Transform the indices back into the original square, remove duplicates
    simplices = np.array(simplices) % n_points
    simplices = np.sort(simplices, axis=1)
    simplices = np.unique(simplices, axis=0)

    # Create the networkx graph
    graph = nx.Graph()

    # Add the edges to the graph
    for simplex in simplices:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            graph.add_edge(
                simplex[i],
                simplex[j],
                weight=np.linalg.norm(diff(points[simplex[i]], points[simplex[j]], lattice_size))
            )
    return graph, points


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
