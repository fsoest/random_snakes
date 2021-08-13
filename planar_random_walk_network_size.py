import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import curve_fit

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import calculate_mean_displacement

n_points_arr = np.arange(3, 150)
t_max = 40
walks = 100

fig, ax = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)

k_list = []
for n_points in n_points_arr:
    planar_graph = PlanarGraph(n_points)
    spl = dict(nx.all_pairs_dijkstra_path_length(planar_graph.graph))

    # Realize random walks, calculate the average
    t_arr, r_arr = calculate_mean_displacement(
        planning_distance=0,
        graph=planar_graph.graph,
        shortest_path_length=spl,
        embedding=planar_graph.embedding,
        t_max=t_max,
        n_walks=walks
    )

    # Fit k * sqrt(D)
    k_fit, covariance = curve_fit(
        f=lambda t, k: k * np.sqrt(t),
        xdata=t_arr,
        ydata=r_arr
    )
    print(f'N = {n_points}, k = {k_fit}')

    # Store k
    k_list.append(k_fit)

    # Plot the curve
    ax[0].plot(t_arr, r_arr, 'x')
    ax[0].plot(t_arr, k_fit * np.sqrt(t_arr), '--')

# plot k(N)
ax[1].plot(n_points_arr, k_list)
plt.show()




