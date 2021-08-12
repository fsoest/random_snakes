import matplotlib.pyplot as plt
import numpy as np

from random_snakes.time_series_average import average_time_series

# Generate dummy data
N = 5
series = []
for i in range(2):
    A = np.random.random() * 5
    B = np.random.random()
    noise = np.random.random(N) * 0.2

    t_arr = np.concatenate([
        [0],
        np.sort(np.random.random(N - 1))
    ])
    x_arr = A * t_arr + B + noise

    series.append(np.stack([t_arr, x_arr], axis=1))

# Compute the interpolation
ip_t_arr, ip_x_arr = average_time_series(series)
print(ip_t_arr.shape)

# Generate a plot
fig, ax = plt.subplots(tight_layout=True)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
for s, label in zip(series, ['A', 'B']):
    ax.plot(*s.T, 'x-', label=f'Time series {label}')
for t in ip_t_arr:
    ax.axvline(t, c='k', ls='--', lw=0.5)
ax.plot(ip_t_arr, ip_x_arr, 'kx-', label=f'Averaged time series')
ax.legend()
fig.savefig('figs/time_series_average.pdf')
plt.show()
