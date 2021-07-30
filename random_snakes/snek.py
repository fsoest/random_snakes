from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import networkx as nx
import numpy as np


def diff(new, old, edge_length):
    """
    Implement the correct distance over boundaries
    """
    dr = np.array(new) - np.array(old)
    dr = np.where(np.abs(dr) > edge_length / 2, -edge_length * np.sign(dr) + dr, dr)
    return dr[0], dr[1]


def select_random_tuple_from_list(tuple_list: List[Tuple]) -> Tuple:
    """
    Return a random tuple from the provided list.
    Necessary because numpy.choice converts lists of tuples into array.
    """
    ind = np.random.choice(len(tuple_list))
    return tuple_list[ind]


def random_snake(g: nx.Graph, d: float, spl: Dict[Any, Dict[Any, float]], lattice_size: int, reps: int = 50,
                 points=None, print_steps: bool = False):
    # Start at a random node
    initial_node = select_random_tuple_from_list(list(g))

    # Initialize node lists and time accumulator
    route = [initial_node]
    plan = []
    steps = []
    t = 0

    while len(route) <= reps:
        if print_steps:
            print(f't = {t:.2f}, route {route}')

        # Populate the plan
        while True:
            last_planned_node = plan[-1] if plan else route[-1]
            neighbours = list(g[last_planned_node])

            # Filter the neighbors
            suitable_neighbours = [
                n for n in neighbours
                if (
                        # This is the fastest route to the node
                        not plan or (np.abs(spl[route[-1]][n] - spl[route[-1]][plan[-1]] - spl[plan[-1]][n]) < 1e-16)
                        # The node is within distance d of the current node
                        and spl[route[-1]][n] <= d
                )
            ]

            if print_steps:
                print('Plan', plan)
                print('Last planned node', last_planned_node)
                print('Neighbors', neighbours)
                print('Suitable neighbors', suitable_neighbours)

            if suitable_neighbours:
                # Add one of the suitable neighbors to the plan
                new_planned_node = select_random_tuple_from_list(suitable_neighbours)
                print(f'Adding {new_planned_node} to plan')
                plan.append(new_planned_node)
            else:
                print('No suitable neighbor!')
                break

        # Perform a step
        if points is not None:
            # An embedding is provided, use it to calculate the distances
            dx, dy = diff(points[plan[0]], points[route[-1]], lattice_size)
        else:
            # The nodes provide also signify their position
            dx, dy = diff(plan[0], route[-1], lattice_size)

        # Calculate time of step, given by 2-norm of dx, dy
        t += np.sqrt(dx ** 2 + dy ** 2)
        # Store the properties of the step
        steps.append({
            'old': route[-1],
            'new': plan[0],
            'dx': dx,
            'dy': dy,
            't': t
        })
        # Move to the next node in the plan
        route.append(plan.pop(0))

        if print_steps:
            print(f'Stepping: {steps[-1]}')
            print()
    return route, steps


def make_r(steps):
    # Generate the time array
    t_arr = np.array([step['t'] for step in steps])

    # Generate the r array by adding up the steps
    dr_arr = np.array([(step['dx'], step['dy']) for step in steps])
    r_arr = np.cumsum(dr_arr, axis=0)
    return r_arr, t_arr
