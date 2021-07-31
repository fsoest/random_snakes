from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from random_snakes.snek import diff


class GraphGenerator:
    def __init__(self, n_points: int, lattice_size: float):
        # Initialize attributes
        self.lattice_size = lattice_size
        self.n_points = n_points
        self.graph = None
        self.embedding = None

        # Generate the graph
        self._generate_graph()

    def _generate_graph(self):
        raise NotImplementedError

    def plot_graph(self, ax: plt.Axes):
        raise NotImplementedError


class PlanarGraph(GraphGenerator):
    def __init__(self, n_points: int, lattice_size: float = 1.):
        self.extended_embedding = None
        self.triangulation = None
        super().__init__(n_points, lattice_size)

    def _generate_graph(self):
        # Randomly draw points from the defined bounds
        self.embedding = np.random.uniform(0, self.lattice_size, (self.n_points, 2))

        # Create empty array to store the periodic extension
        self.extended_embedding = np.zeros((9 * self.n_points, 2))

        # Extend the lattice in a 3x3 array
        a = [0, 1, -1]
        k = 0
        for i in a:
            for j in a:
                shift = self.embedding + np.array([i, j]) * self.lattice_size
                self.extended_embedding[k * self.n_points: (k + 1) * self.n_points] = shift
                k += 1

        # Calculate the Delaunay triangulation over the periodic extension
        self.triangulation = Delaunay(self.extended_embedding)

        # Determine the triangles that contain at least one node in the original lattice
        simplices = [
            simplex for simplex in self.triangulation.simplices
            if min(simplex) < self.n_points
        ]

        # Transform the indices back into the original square, remove duplicates
        simplices = np.array(simplices) % self.n_points
        simplices = np.sort(simplices, axis=1)
        simplices = np.unique(simplices, axis=0)

        # Create the networkx graph
        self.graph = nx.Graph()

        # Add the edges to the graph
        for simplex in simplices:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                self.graph.add_edge(
                    simplex[i],
                    simplex[j],
                    weight=np.linalg.norm(diff(self.embedding[simplex[i]], self.embedding[simplex[j]], self.lattice_size))
                )

    def plot_graph(self, ax: plt.Axes):
        # Set the axis limits
        ax.axis('equal')

        # Plot the edges:
        # Create an array with shape ([start, end], n_edges, [x, y])
        edge_arr = np.stack([
            np.stack([
                self.extended_embedding[simplex[i]],
                self.extended_embedding[simplex[j]]
            ])
            for i, j in [(0, 1), (1, 2), (2, 0)]
            for simplex in self.triangulation.simplices
        ], axis=1)

        # Plot the array
        ax.plot(
            edge_arr[:, :, 0],
            edge_arr[:, :, 1],
            c='gray',
            ls='-',
        )

        # Draw lines where the graph repeats
        for i in [0, 1]:
            ax.axhline(i * self.lattice_size, c='k')
            ax.axvline(i * self.lattice_size, c='k')