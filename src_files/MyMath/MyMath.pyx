import math
import cython
import numpy as np

from libc.math cimport exp

# constants
cdef double LOOKUP_DEGREE_RESOLUTION = 0.05  # 0.05 degree resolution
cdef double LOOKUP_EXP_RESOLUTION = 0.001
cdef double LOOKUP_EXP_BOUNDS = 10

# important parts
radians = np.array(
    range(0, math.ceil(360 / LOOKUP_DEGREE_RESOLUTION) + 1),
) * LOOKUP_DEGREE_RESOLUTION * math.pi / 180
exp_range = np.arange(-LOOKUP_EXP_BOUNDS, LOOKUP_EXP_BOUNDS + LOOKUP_EXP_RESOLUTION, LOOKUP_EXP_RESOLUTION)

cdef double[::1] lookup_sin_degrees = np.sin(radians).astype(np.float64)
cdef double[::1] lookup_cos_degrees = np.cos(radians).astype(np.float64)
cdef double[::1] lookup_exp_array = np.exp(exp_range).astype(np.float64)
cdef double exp_lookup_shift = LOOKUP_EXP_BOUNDS / LOOKUP_EXP_RESOLUTION

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
