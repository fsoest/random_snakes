from typing import Dict, Any
import numpy as np
import networkx as nx


def diff(new, old, D):
    """
    Implement the correct distance over boundaries
    :param a:
    :param b:
    :return:
    """
    x = (new[0] - old[0]) % D
    y = (new[1] - old[1]) % D
    # if new[0] < old[0]:
    #     x = min(np.abs(new[0] - old[0]), np.abs(new[0] - old[0] + D))
    # else:
    #     x = max(new[0] - old[0], new[0] - old[0] - D)
    # if new[1] < old[1]:
    #     y = min(np.abs(new[1] - old[1]), np.abs(new[1] - old[1] + D))
    # else:
    #     y = max(new[1] - old[1], new[1] - old[1] - D)
    return (x, y)


def random_snake(g: nx.Graph, d: float, spl: Dict[Any, Dict[Any, float]], lattice_size: int, reps: int = 50):
    # Calculate shortest path lengths

    initial_node_index = np.random.choice(len(g))
    initial_node = list(g)[initial_node_index]

    route = []
    plan = []
    steps = []
    # First step is random
    n_0 = list(g[initial_node])
    plan_index = np.random.choice(np.arange(len(n_0)))
    plan.append(n_0[plan_index])
    route.append(initial_node)
    t = 0

    while len(route) < reps:
        # Get neighbours of last node in plan or x_0
        try:
            neighbours = list(g[plan[-1]])
        except:
            neighbours = list(g[route[-1]])
            # print(plan)
            # print(route)
        # # Check suitability of neighbours
        suitables = []
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
            dx, dy = diff(plan[0], route[-1], lattice_size)
            # Calculate time of step, given by 2-norm of dx, dy
            t += np.sqrt(dx ** 2 + dy ** 2)
            step = {
                'old': route[-1],
                'new': plan[0],
                'dx': dx,
                'dy': dy,
                't': t
            }
            steps.append(step)
            route.append(plan.pop(0))
    return route, steps


def make_r(steps):
    pass
    # r = np.