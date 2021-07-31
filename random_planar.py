import networkx as nx
import numpy as np

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r
from random_snakes.time_series_average import average_time_series
import matplotlib.pyplot as plt


if __name__ == '__main__':
    seed = np.random.randint(0, 255)
    np.random.seed(seed)
    print(f'Seed: {seed}')

    n_points = 100
    t_max = 100
    walks = 5
    d_arr = np.linspace(0, 0.5, 5)


    planar_graph = PlanarGraph(n_points)
    G = planar_graph.graph
    points = planar_graph.embedding
    spl = dict(nx.all_pairs_dijkstra_path_length(G))

    mean_displacement_series = []
    for d in d_arr:
        displacement_series = []
        for i in range(walks):
            route, steps = random_snake(G, d, spl, lattice_size=1, points=points, t_max=t_max, verbose=False)
            r, t = make_r(steps)
            r_abs = np.sqrt(np.sum(r ** 2, axis=1))
            displacement_series.append(np.stack([t, r_abs], axis=1))
        mean_displacement_series.append(average_time_series(displacement_series))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, tight_layout=True)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\langle \ \|\| r(t) \|\|_2 \ \\rangle$ ')

    for d, (t, r) in zip(d_arr, mean_displacement_series):
        ax.plot(t, r, label=f'{d}')
    ax.legend()
    plt.show()
