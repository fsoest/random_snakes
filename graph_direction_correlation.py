import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator

from random_snakes.graph_generators import LatticeGraph
from random_snakes.snek import make_r
from random_snakes.snek import random_snake

# Number of walks to average over
n_walks = 1000
# Array of planning distance values
d_arr = np.arange(1, 6)
print(d_arr)
# Maximum simulation time
t_max = 100
min_length = 50

# Initialize the graph
graph = LatticeGraph(lattice_size=10)
spl = dict(nx.all_pairs_shortest_path_length(graph.graph))

# Create a figure
fig, ax = plt.subplots(tight_layout=True)

for d in d_arr:
    print(d)
    current_series = []
    for i in range(n_walks):
        # Perform the walk
        route, steps = random_snake(
            graph.graph, d, spl,
            lattice_size=10,
            t_max=t_max,
        )

        # Create an array of the step direction and determine the index of the t_0 step
        r_arr, t_arr = make_r(steps)
        delta_r_arr = r_arr[1:] - r_arr[:-1]
        for t_0_ind in range(0, delta_r_arr.shape[0] - min_length):
            # Compute the angle between the initial step and future steps
            delta_r_0 = delta_r_arr[t_0_ind]
            delta_r_tau_arr = delta_r_arr[t_0_ind + 1:]
            angle_arr = np.arccos(
                np.sum(delta_r_0 * delta_r_tau_arr, axis=1)
                / np.linalg.norm(delta_r_0)
                / np.linalg.norm(delta_r_tau_arr, axis=1)
            )
            current_series.append(angle_arr)

    avg_angle_arr = np.mean(np.stack([
        arr[:min_length]
        for arr in current_series
    ], axis=1), axis=1)
    ax.plot(avg_angle_arr / np.pi, label=f'd = {d}')

ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\phi(\tau)$')
ax.set_ylim(0, 0.7)
ax.grid()
ax.yaxis.set_major_formatter(FormatStrFormatter(r'%g $\pi$'))
ax.yaxis.set_major_locator(MultipleLocator(base=0.5))
ax.legend()
fig.savefig('figs/graph_direction_correlation.pdf')
plt.show()
