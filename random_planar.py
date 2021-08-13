from functools import partial
from multiprocessing import Pool
from typing import Tuple

import networkx as nx
import numpy as np
from scipy.interpolate import interp1d

from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import random_snake, make_r
from random_snakes.time_series_average import average_time_series
import matplotlib.pyplot as plt


def calculate_mean_displacement(
        planning_distance: float,
        graph: nx.Graph,
        n_walks: int,
        shortest_path_length: dict,
        embedding: np.ndarray,
        t_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    displacement_series = []
    for i in range(n_walks):
        route, steps = random_snake(
            graph, planning_distance, shortest_path_length,
            lattice_size=1,
            points=embedding,
            t_max=t_max,
            verbose=False
        )
        r, t = make_r(steps)
        r_abs = np.sqrt(np.sum(r ** 2, axis=1))
        displacement_series.append(np.stack([t, r_abs], axis=1))
    return average_time_series(displacement_series)


if __name__ == '__main__':
    seed = np.random.randint(0, 255)
    np.random.seed(seed)
    print(f'Seed: {seed}')

    n_points = 100
    t_max = 40
    walks = 75
    d_arr = np.linspace(0, 0.5, 60)

    planar_graph = PlanarGraph(n_points)
    G = planar_graph.graph
    points = planar_graph.embedding
    spl = dict(nx.all_pairs_dijkstra_path_length(G))

    mean_displacement_partial = partial(
        calculate_mean_displacement,
        graph=G,
        n_walks=walks,
        shortest_path_length=spl,
        embedding=points,
        t_max=t_max
    )
    with Pool() as p:
        mean_displacement_series = p.map(mean_displacement_partial, d_arr)

    plotting_t_arr = np.linspace(0, t_max, 2000)

    interpolated_mean_displacement = np.array([
        interp1d(t, r)(plotting_t_arr)
        for t, r in mean_displacement_series
    ])

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), tight_layout=True)

    for d, (t, r) in zip(d_arr, mean_displacement_series):
        axes[0].plot(t, r, label=f'{d}')

    axes[1].imshow(
        interpolated_mean_displacement,
        extent=[
            min(plotting_t_arr),
            max(plotting_t_arr),
            max(d_arr),
            min(d_arr)
        ],
        aspect='auto',
        cmap='gray',
        interpolation='nearest'
    )
    plt.show()
