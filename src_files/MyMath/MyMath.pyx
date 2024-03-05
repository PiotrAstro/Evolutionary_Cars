import math
import cython
import numpy as np

from libc.math cimport exp
from scipy.stats import norm

from src_files.MyMath.cython_debug_helper import cython_debug_call

# constants
cdef double LOOKUP_DEGREE_RESOLUTION = 0.05  # 0.05 degree resolution
cdef double LOOKUP_EXP_RESOLUTION = 0.001
cdef double LOOKUP_EXP_BOUNDS = 8
cdef double LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION = 0.001

# important parts
radians = np.array(
    range(0, math.ceil(360 / LOOKUP_DEGREE_RESOLUTION) + 1),
) * LOOKUP_DEGREE_RESOLUTION * math.pi / 180
exp_range = np.arange(-LOOKUP_EXP_BOUNDS, LOOKUP_EXP_BOUNDS + LOOKUP_EXP_RESOLUTION, LOOKUP_EXP_RESOLUTION)
normal_distribution_range = np.arange(0.0, 1.0 + LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION, LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION)
normal_distribution_range[0] = 0.0000001
normal_distribution_range[-1] = 0.9999999

cdef double[::1] lookup_sin_degrees = np.sin(radians).astype(np.float64)
cdef double[::1] lookup_cos_degrees = np.cos(radians).astype(np.float64)
cdef double[::1] lookup_exp_array = np.exp(exp_range).astype(np.float64)
cdef double[::1] lookup_normal_distribution_array = np.array(norm.ppf(normal_distribution_range)).astype(np.float64)
cdef double exp_lookup_shift = LOOKUP_EXP_BOUNDS / LOOKUP_EXP_RESOLUTION


# cython_debug_call(
#     {
#         "radians": radians,
#         "exp_lookup_shift": exp_lookup_shift,
#         "lookup_sin_degrees": np.array(lookup_sin_degrees),
#         "lookup_cos_degrees": np.array(lookup_cos_degrees),
#         "lookup_exp_array": np.array(lookup_exp_array),
#         "lookup_exp_4000": lookup_exp_array[4000],
#         "lookup_normal_distribution_array": np.array(lookup_normal_distribution_array),
#         "lookup_normal_distribution_range": normal_distribution_range,
#     }
# )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double lookup_exp(double value) noexcept nogil:
    if value >= LOOKUP_EXP_BOUNDS or value <= -LOOKUP_EXP_BOUNDS:
        return exp(value)
    else:
        return lookup_exp_array[<int>(value / LOOKUP_EXP_RESOLUTION + exp_lookup_shift + 0.5)]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double degree_sin(double degree) noexcept nogil:
    """
    Return the sin of the given degree.
    degree: double in any range (will be converted to [0, 360))
    I do not check the input range because of performance!
    """
    degree = degree % 360
    return lookup_sin_degrees[<int>(degree / LOOKUP_DEGREE_RESOLUTION + 0.5)]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double degree_cos(double degree) noexcept nogil:
    """
    Return the cos of the given degree.
    degree: double in any range (will be converted to [0, 360))
    I do not check the input range because of performance!
    """
    degree = degree % 360
    return lookup_cos_degrees[<int>(degree / LOOKUP_DEGREE_RESOLUTION + 0.5)]


cdef inline int round_to_int(double value) noexcept nogil:
    """
    Round the given value to the nearest integer.
    """
    return <int>(value + 0.5)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double lookup_normal_distribution(double value) noexcept nogil:
    """
    Return value of inverse normal distribution function for the given value, so you put random number between 0 and 1 and get one value from ~N(0, 1).
    :param value: double in range [0, 1)
    """
    return lookup_normal_distribution_array[<int>(value / LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION)]

@cython.boundscheck(False)
@cython.wraparound(False)
def mutate_array_scaled(array_modified: np.ndarray, scale: float, threshold: float) -> None:
    """
    Mutate the given array by adding random values to each element. Works in place.
    algorithm:

    for value in array_modified:
        scale_here = scale
        if abs(value) > treshold:
            scale_here *= abs(value)
        else:
            scale_here *= treshold
        value += lookup_normal_distribution(uniform(0, 1)) * scale_here

    Random values are taken from numpy, so one can take care of seed globally.

    :param array_modified: 1d or 2d numpy array, modified in place, float32
    :param scale: basic scale, like sigma of normal distribution, scales everything, e.g. 0.1
    :param threshold: threshold for scaling, if value is bigger than threshold, scale is multiplied by value, else by threshold
    :return: None, modifies the array in place
    """
    # initial_array = np.array(array_modified, copy=True, dtype=np.float32)

    cdef float[:, ::1] array_modified_view
    cdef double[:, ::1] random_values
    cdef int i, j
    cdef int rows
    cdef int cols
    cdef double scale_here
    cdef double threshold_here = threshold
    cdef double tmp_modified_value

    if len(array_modified.shape) == 1:
        array_modified_view = array_modified[None, :]
        random_values = np.random.uniform(0, 1, (1, array_modified.shape[0]))
    else:
        array_modified_view = array_modified
        random_values = np.random.uniform(0, 1, array_modified.shape)
    rows = array_modified_view.shape[0]
    cols = array_modified_view.shape[1]

    with nogil:
        for i in range(rows):
            for j in range(cols):
                tmp_modified_value = array_modified_view[i, j]
                scale_here = scale
                if tmp_modified_value > threshold_here:
                    scale_here *= tmp_modified_value
                elif tmp_modified_value < -threshold_here:
                    scale_here *= -tmp_modified_value
                else:
                    scale_here *= threshold_here

                array_modified_view[i, j] = tmp_modified_value + lookup_normal_distribution(random_values[i, j]) * scale_here

    # values_from_normal_distribution = np.zeros_like(random_values)
    # for i in range(random_values.shape[0]):
    #     for j in range(random_values.shape[1]):
    #         values_from_normal_distribution[i, j] = lookup_normal_distribution(random_values[i, j])
    # cython_debug_call(
    #     {
    #         "initial_array": initial_array,
    #         "array_modified": array_modified,
    #         "random_values": np.array(random_values),
    #         "values_from_normal_distribution": np.array(values_from_normal_distribution),
    #         "scale": scale,
    #         "threshold_here": threshold_here,
    #     }
    # )
