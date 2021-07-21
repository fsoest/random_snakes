import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def diff(a, b):
    return (a[0] - b[0], a[1] - b[1])


def random_snake(g, d, spl, reps=50):
    # Calculate shortest path lengths

    initial_node_index = np.random.choice(np.arange(len(g)))
    initial_node = list(g)[initial_node_index]

    route = []
    plan = []

    # First step is random
    n_0 = list(g[initial_node])
    plan_index = np.random.choice(np.arange(len(n_0)))
    plan.append(n_0[plan_index])
    route.append(initial_node)

    while len(route) < reps:
        # Get neighbours of last node in plan or x_0
        try:
            neighbours = list(g[plan[-1]])
        except:
            neighbours = list(g[initial_node])
            # print(plan)
            # print(route)
        # # Check suitability of neighbours
        suitables = []
        # print(_)
        # print('Plan', plan)
        # print('route', route)
        # print('neighbours', neighbours)
        for n in neighbours:
            # Check for empty plan
            if len(plan) != 0:
                if spl[route[-1]][n] == spl[route[-1]][plan[-1]] + spl[plan[-1]][n] and spl[route[-1]][n] <= d:
                    suitables.append(n)
            else:
                suitables = list(g[route[-1]])
        # print('suitables', suitables)
        # Choose random neighbour from list of suitable neighbours if list nonempty
        if len(suitables) > 0:
            n_index = np.random.choice(np.arange(len(suitables)))
            plan.append(suitables[n_index])
        else:
            # Perform a step
            # dx, dy =
            route.append(plan[0])
            plan.remove(plan[0])
    return route


if __name__ == '__main__':
    # np.random.seed(42)
    lattice_size = 50
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)
    spl = dict(nx.all_pairs_shortest_path_length(square_lattice))
    route = random_snake(square_lattice, 4, spl, reps=100)

    xx, yy = np.meshgrid(np.arange(lattice_size), np.arange(lattice_size))

    xs = [x[0] for x in route]
    ys = [x[1] for x in route]

    print(route)
    plt.scatter(xx, yy, c='k', marker='.', alpha=0.1)
    plt.scatter(xs[0], ys[0], c='b')
    plt.scatter(xs[1:], ys[1:], c='r')
    plt.show()
