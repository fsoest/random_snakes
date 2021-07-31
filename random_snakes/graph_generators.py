from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from random_snakes.snek import diff


class GraphGenerator:
    def __init__(self, n_points: int, lattice_size: float):
        self.lattice_size = lattice_size
        self.n_points = n_points

        self.graph, self.embedding = self._generate_graph()

    def _generate_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        raise NotImplementedError

    def plot_graph(self, ax: plt.Axes):
        raise NotImplementedError


class PlanarGraph(GraphGenerator):
    def __init__(self, n_points: int, lattice_size: float):
        super().__init__(n_points, lattice_size)

    def _generate_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        # Randomly draw points from the defined bounds
        points = np.random.uniform(0, self.lattice_size, (self.n_points, 2))

        # Create empty array to store the periodic extension
        periodic = np.zeros((9 * self.n_points, 2))

        # Extend the lattice in a 3x3 array
        a = [0, 1, -1]
        k = 0
        for i in a:
            for j in a:
                shift = points + np.array([i, j]) * self.lattice_size
                periodic[k * self.n_points: (k + 1) * self.n_points] = shift
                k += 1

        # Calculate the Delaunay triangulation over the periodic extension
        tri = Delaunay(periodic)

        # Determine the triangles that contain at least one node in the original lattice
        simplices = [
            simplex for simplex in tri.simplices
            if min(simplex) < self.n_points
        ]

        # Transform the indices back into the original square, remove duplicates
        simplices = np.array(simplices) % self.n_points
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
                    weight=np.linalg.norm(diff(points[simplex[i]], points[simplex[j]], self.lattice_size))
                )
        return graph, points
