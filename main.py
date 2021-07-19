import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def random_snake(g, d):
    # Calculate shortest path lengths
    spl = dict(nx.all_pairs_shortest_path_length(g))

    initial_node_index = np.random.choice(np.arange(len(g)))
    initial_node = list(g)[initial_node_index]
    current_node = initial_node
    route = []
    route.append(current_node)
    # Plan list, only includes nodes that are yet to be visited
    plan = []
    while True:
        neighbours = list(square_lattice[current_node])
        for neighbour in neighbours:
            # Check suitability of neighbour with Δ-inequality
            if (spl[route[-1]][neighbour] != spl[route[-1]][current_node] + spl[current_node][neighbour]) or \
                    (spl[route[-1]][neighbour] > d):
                neighbours.remove(neighbour)
        if len(neighbours) == 0:
            # TODO: Implement step
            # Add first step in plan to route
            route.append(plan[0])
            plan.remove(plan[0])
        else:
            # Randomly choose next node from suitable neighbours
            next_index = np.random.choice(np.arange(len(neighbours)))
            next_node = neighbours[next_index]
            plan.append(next_node)
            current_node = next_node

if __name__ == '__main__':
    # Create square lattice
    lattice_size = 5
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)
