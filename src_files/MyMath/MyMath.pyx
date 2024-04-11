import math
import cython
import numpy as np

from libc.math cimport exp, log, sqrt
from scipy.stats import norm

from src_files.MyMath.cython_debug_helper import cython_debug_call

# constants
cdef double LOOKUP_DEGREE_RESOLUTION = 0.05  # 0.05 degree resolution
cdef double LOOKUP_EXP_RESOLUTION = 0.001
cdef double LOOKUP_EXP_BOUNDS = 10
cdef double TANH_SIGMOID_CLIP = 20
cdef double LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION = 0.001
cdef double LOOKUP_LN_BOUNDS = 10
cdef double LOOKUP_LN_RESOLUTION = 0.001

# important parts
radians = np.array(
    range(0, math.ceil(360 / LOOKUP_DEGREE_RESOLUTION) + 1),
) * LOOKUP_DEGREE_RESOLUTION * math.pi / 180
exp_range = np.arange(-LOOKUP_EXP_BOUNDS, LOOKUP_EXP_BOUNDS + LOOKUP_EXP_RESOLUTION, LOOKUP_EXP_RESOLUTION)
ln_range = np.arange(0, LOOKUP_LN_BOUNDS + LOOKUP_LN_RESOLUTION, LOOKUP_LN_RESOLUTION)
ln_range[0] = 0.00000001
normal_distribution_range = np.arange(0.0, 1.0 + LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION, LOOKUP_NORMAL_DISTRIBUTION_RESOLUTION)
normal_distribution_range[0] = 0.0000001
normal_distribution_range[-1] = 0.9999999

cdef double[::1] lookup_sin_degrees = np.sin(radians).astype(np.float64)
cdef double[::1] lookup_cos_degrees = np.cos(radians).astype(np.float64)
cdef double[::1] lookup_exp_array = np.exp(exp_range).astype(np.float64)
cdef double[::1] lookup_normal_distribution_array = np.array(norm.ppf(normal_distribution_range)).astype(np.float64)
cdef double[::1] lookup_ln_array = np.log(ln_range).astype(np.float64)
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
cdef double lookup_ln(double value) noexcept nogil:
    if value <= 0:
        return -1000
    elif value >= LOOKUP_LN_BOUNDS:
        return log(value)
    else:
        return lookup_ln_array[<int>(value / LOOKUP_LN_RESOLUTION + 0.5)]

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

cdef inline double tanh(double value) noexcept nogil:
    """
    Return the hyperbolic tangent of the given value.
    """
    cdef double exp_2x
    if value > TANH_SIGMOID_CLIP:
        return 1.0
    elif value < -TANH_SIGMOID_CLIP:
        return -1.0
    else:
        exp_2x = lookup_exp(2.0 * value)
        return (exp_2x - 1.0) / (exp_2x + 1.0)

cdef inline double sigmoid(double value) noexcept nogil:
    """
    Return the sigmoid of the given value.
    """
    if value > TANH_SIGMOID_CLIP:
        return 1.0
    elif value < -TANH_SIGMOID_CLIP:
        return 0.0
    else:
        return 1.0 / (1.0 + lookup_exp(-value))

cdef inline float float_abs(float value) noexcept nogil:
    """
    Return the absolute value of the given float.
    """
    if value < 0:
        return -value
    else:
        return value


@cython.boundscheck(False)
@cython.wraparound(False)
def safe_mutate_inplace(array_modified: np.ndarray,
                        mutation_factor: float,
                        scales: np.ndarray,
                        min_scale: float,
                        max_scale: float,
                        L1_norm: float,
                        L2_norm: float) -> None:
    """
    Mutate the given array by adding random values to each element. Works in place.
    Random values are taken from numpy, so one can take care of seed globally.
    :param array_modified: 2d numpy array, modified in place (float32)
    :param scales: 2d numpy array, scales for each element (float32)
    :param min_scale: minimum scale for each element
    :param max_scale: maximum scale for each element
    :param L1_norm: L1 norm of the scales
    :param L2_norm: L2 norm of the scales
    :return: None, modifies the array in place
    """
    cdef float[:, ::1] array_modified_view
    cdef double[:, ::1] random_values
    cdef float[:, ::1] scales_view
    cdef int i, j
    cdef int rows
    cdef int cols
    cdef float mutation_factor_here = mutation_factor
    cdef float min_scale_here = min_scale
    cdef float max_scale_here = max_scale
    cdef float L1_norm_here = L1_norm
    cdef float L2_norm_here = L2_norm
    cdef float L1_norm_tmp
    cdef float scale_here
    cdef float tmp_modified_value

    if len(array_modified.shape) == 1:
        array_modified_view = array_modified[None, :]
        scales_view = scales[None, :]
        random_values = np.random.uniform(0, 1, (1, array_modified.shape[0]))
    else:
        array_modified_view = array_modified
        scales_view = scales
        random_values = np.random.uniform(0, 1, array_modified.shape)

    with nogil:
        rows = array_modified_view.shape[0]
        cols = array_modified_view.shape[1]
        for i in range(rows):
            for j in range(cols):
                tmp_modified_value = array_modified_view[i, j]
                scale_here = scales_view[i, j]
                if scale_here < min_scale_here:
                    scale_here = min_scale_here
                elif scale_here > max_scale_here:
                    scale_here = max_scale_here
                L1_norm_tmp = L1_norm_here
                if tmp_modified_value < 0:
                    L1_norm_tmp *= -1
                array_modified_view[i, j] += lookup_normal_distribution(random_values[i, j]) * mutation_factor_here / scale_here - L1_norm_tmp - L2_norm_here * tmp_modified_value

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def mutate_array_scaled(array_modified: np.ndarray, scale: float, threshold: float) -> None:
#     """
#     Mutate the given array by adding random values to each element. Works in place.
#     Random values are taken from numpy, so one can take care of seed globally.
#
#     :param array_modified: 1d or 2d numpy array, modified in place, float32
#     :param scale: basic scale, like sigma of normal distribution, scales everything, e.g. 0.1
#     :param threshold: threshold for scaling, if value is bigger than threshold, scale is multiplied by value, else by threshold
#     :return: None, modifies the array in place
#     """
#     # initial_array = np.array(array_modified, copy=True, dtype=np.float32)
#
#     cdef float[:, ::1] array_modified_view
#     cdef double[:, ::1] random_values
#     cdef int i, j
#     cdef int rows
#     cdef int cols
#     cdef double scale_here
#     cdef double mean
#     cdef double std_dev
#     cdef double c_scale = scale
#     cdef double threshold_here = threshold
#     cdef double tmp_modified_value
#
#     if len(array_modified.shape) == 1:
#         array_modified_view = array_modified[None, :]
#         random_values = np.random.uniform(0, 1, (1, array_modified.shape[0]))
#     else:
#         array_modified_view = array_modified
#         random_values = np.random.uniform(0, 1, array_modified.shape)
#     rows = array_modified_view.shape[0]
#     cols = array_modified_view.shape[1]
#
#     with nogil:
#         for j in range(cols):
#             mean = 0.0
#             std_dev = 0.0
#             for i in range(rows):
#                 tmp_modified_value = array_modified_view[i, j]
#                 std_dev += tmp_modified_value**2
#                 mean += tmp_modified_value
#             mean /= rows
#             std_dev = sqrt(std_dev / rows - mean**2)
#
#             for i in range(rows):
#                 tmp_modified_value = array_modified_view[i, j]
#
#                 if std_dev == 0:
#                     std_dev = 1
#                 tmp_modified_value /= std_dev
#
#                 if tmp_modified_value < 0:
#                     tmp_modified_value = -tmp_modified_value
#
#                 tmp_modified_value = 3 * (tmp_modified_value - 1)
#                 scale_here = 1 + threshold_here * sigmoid(tmp_modified_value)
#                 scale_here *= c_scale
#                 array_modified_view[i, j] += lookup_normal_distribution(random_values[i, j]) * scale_here
#
#         # for i in range(rows):
#         #     for j in range(cols):
#                 # original, normal threshold
#                 # tmp_modified_value = array_modified_view[i, j]
#                 # scale_here = c_scale
#                 # if tmp_modified_value > threshold_here:
#                 #     scale_here *= tmp_modified_value
#                 # elif tmp_modified_value < -threshold_here:
#                 #     scale_here *= -tmp_modified_value
#                 # else:
#                 #     scale_here *= threshold_here
#                 #
#                 # array_modified_view[i, j] = tmp_modified_value + lookup_normal_distribution(random_values[i, j]) * scale_here
#
#     # values_scale_here = np.zeros_like(random_values)
#     # for j in range(cols):
#     #     mean = 0.0
#     #     std_dev = 0.0
#     #     for i in range(rows):
#     #         tmp_modified_value = array_modified_view[i, j]
#     #         std_dev += tmp_modified_value ** 2
#     #         mean += tmp_modified_value
#     #     mean /= rows
#     #     std_dev = sqrt(std_dev / rows - mean ** 2)
#     #
#     #     for i in range(rows):
#     #         tmp_modified_value = array_modified_view[i, j]
#     #         tmp_modified_value /= std_dev
#     #
#     #         if tmp_modified_value < 0:
#     #             tmp_modified_value = -tmp_modified_value
#     #
#     #         tmp_modified_value = 3 * (tmp_modified_value - 1)
#     #         scale_here = 1 + threshold_here * sigmoid(tmp_modified_value)
#     #         values_scale_here[i, j] = scale_here
#     # cython_debug_call(
#     #     {
#     #         "initial_array": initial_array,
#     #         "array_modified": array_modified,
#     #         "random_values": np.array(random_values),
#     #         "values_scale_here": np.array(values_scale_here),
#     #         "scale": scale,
#     #         "threshold_here": threshold_here,
#     #     }
#     # )
