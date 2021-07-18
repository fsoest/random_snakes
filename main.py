import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Create square lattice
    lattice_size = 5
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)

    print(square_lattice.nodes())