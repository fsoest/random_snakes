import numpy as np
from random_snakes.graph_generators import PlanarGraph
from random_snakes.snek import make_r, random_snake
import matplotlib.pyplot as plt
from random_snakes.time_series_average import average_time_series
import networkx as nx


def main():
    np.random.seed(42)
    ds = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    t_max = 50
    n_graphs = 35
    n_walks = 35

    avg = []

    for d in ds:
        displacement_series = []
        for j in range(n_graphs):
            G = PlanarGraph(100)
            spl = dict(nx.all_pairs_dijkstra_path_length(G.graph))
            for k in range(n_walks):
                _, steps = random_snake(G.graph, d, spl, lattice_size=1, points=G.embedding, t_max=t_max, verbose=False)
                r, t = make_r(steps)
                r_abs = np.linalg.norm(r, ord=2, axis=1)
                displacement_series.append(np.stack([t, r_abs], axis=1))
        avg.append(average_time_series(displacement_series))

    fig, ax = plt.subplots()
    for i, d in enumerate(ds):
        ax.plot(avg[i][0], avg[i][1], label=d)
    t = np.linspace(0, t_max)

    s = 2.3

    plot_max = np.amax(avg[-1][1])
    plot_max *= 1.1
    plot_max = int(plot_max + 4)


    ax.plot([0, plot_max], [0, plot_max], 'k--', label='t')
    ax.plot(t, np.sqrt(t)/10, 'k-.', label='$\sqrt{t} / 10$')
    plt.xlabel('$t$')
    plt.ylabel('$\mathcal{D}_2(t)$ ')

    ax.legend(title='d')
    plt.loglog()
    plt.savefig('figures/planar_msd.pdf')
    plt.show()
    

if __name__ == '__main__':
    main()
