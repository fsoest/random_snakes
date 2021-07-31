from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GraphGenerator:
    def __init__(self, lattice_size: float):
        self.lattice_size = lattice_size

        self.graph, self.embedding = self._generate_graph()

    def _generate_graph(self) -> Tuple[nx.Graph, np.ndarray]:
        raise NotImplementedError

    def plot_graph(self, ax: plt.Axes):
        raise NotImplementedError
