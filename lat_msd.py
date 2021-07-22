import matplotlib.pyplot as plt
from snek import *
from save import *


if __name__ == '__main__':
    np.random.seed(42)
    ds = [1, 2, 4, 8, 16]
    walks = 100
    walk_length = 100
    msd = np.zeros((len(ds), walks, walk_length))

    # Initialise lattice
    lattice_size = 50
    square_lattice = nx.grid_graph((lattice_size, lattice_size), periodic=True)
    # spl = dict(nx.all_pairs_shortest_path_length(square_lattice))
    spl = load_obj('spl_50')
    # save_obj(spl, 'spl_50')
    print('SPL done!')
    for i, d in enumerate(ds):
        for j in range(walks):
            route, steps = random_snake(square_lattice, d, spl, lattice_size, reps=walk_length)
            msd[i, j] = np.linalg.norm(make_r(steps), ord=1, axis=1)
        plt.plot(np.mean(msd[i], axis=0), label=d)
    plt.legend(title='$d$')
    plt.xlabel('$t$')
    plt.ylabel('$\langle \ \|\| r(t) \|\|_1 \ \\rangle$ ')
    plt.show()