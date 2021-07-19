import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def random_snake(g, d):
    # Calculate shortest path lengths
    spl = dict(nx.all_pairs_shortest_path_length(g))

    initial_node_index = np.random.choice(np.arange(len(g)))
    initial_node = list(g)[initial_node_index]

    route = []
    plan = []

    # First step is random
    n_0 = list(g[initial_node])
    plan_index = np.random.choice(np.arange(len(n_0)))
    plan.append(list(g)[plan_index])

    route.append(initial_node)

    for _ in range(10):
        # Get neighbours of last node in plan or x_0
        try:
            neighbours = list(g[plan[-1]])
        except:
            neighbours = list(g[initial_node])
        # Check suitability of neighbours
        for n in neighbours:
            if spl[route[-1]][n] != spl[route[-1]][plan[-1]] + spl[plan[-1]][n] or spl[route[-1]][n] > d:
                neighbours.remove(n)
        # Choose random neighbour from list of suitable neighbours if list nonempty
        if len(neighbours) > 0:
            n_index = np.random.choice(np.arange(len(neighbours)))
            plan.append(neighbours[n_index])
        else:
            route.append(plan[0])
            plan.remove(plan[0])
        print(route)
        print(plan)


if __name__ == '__main__':
    lattice_size = 5
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)

    random_snake(square_lattice, 2)