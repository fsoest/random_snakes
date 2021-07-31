from typing import List
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def average_time_series(series: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Generate union of all time values
    t_arr = np.concatenate([s[:, 0] for s in series])
    t_arr = np.unique(t_arr)
    t_arr = np.sort(t_arr)

    # Generate linear interpolation arrays
    interpolated_array = np.stack([
        interp1d(s[:, 0], s[:, 1])(t_arr)
        for s in series
    ])

    # Return the time values and the mean of the array
    return t_arr, np.mean(interpolated_array, axis=0)