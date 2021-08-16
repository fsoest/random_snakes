import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from random_snakes.graph_generators import LatticeGraph

from random_snakes.snek import make_r
from random_snakes.snek import random_snake
from random_snakes.time_series_average import average_time_series


# Number of walks to average over
n_walks = 1000
# Array of planning distance values
d_arr = np.arange(0, 6)
print(d_arr)
# Maximum simulation time
t_max = 100
# Starting point of the correlation measurement
t_0 = 2

# Initialize the graph
graph = LatticeGraph(lattice_size=10)
spl = dict(nx.all_pairs_shortest_path_length(graph.graph))

correlation_series = []
for d in d_arr:
    print(d)
    current_series = []
    for i in range(n_walks):
        # Perform the walk
        route, steps = random_snake(
            graph.graph, d, spl,
            lattice_size=10,
            t_max=t_max,
            #points=graph.embedding
        )

        # Calculate an array of the step direction and the
        r_arr, t_arr = make_r(steps)
        delta_r_arr = r_arr[1:] - r_arr[:-1]
        t_0_ind = np.argwhere(t_arr >= t_0)[0].item()

        # Compute the angle between the initial step and future steps
        delta_r_0 = delta_r_arr[t_0_ind]
        delta_r_tau_arr = delta_r_arr[t_0_ind + 1:]
        angle_arr = np.arccos(
            np.sum(delta_r_0 * delta_r_tau_arr, axis=1)
            / np.linalg.norm(delta_r_0)
            / np.linalg.norm(delta_r_tau_arr, axis=1)
        )
        current_series.append(np.stack([
            np.arange(angle_arr.shape[0]),
            angle_arr
        ], axis=1))

    t_arr, avg_angle_arr = average_time_series(current_series)
    plt.plot(avg_angle_arr, label=d)

plt.xlabel(r'$\tau$')
plt.ylabel(r'$\angle \ \vec{r}(t_0) \ \vec{r}(t_0 + \tau)$')
plt.legend()
plt.savefig('figs/graph_direction_correlation.pdf', bbox_inches='tight')
plt.show()
