import math
from typing import Tuple, List, Union, Any

import cython
import numpy as np
from numpy import ndim
from src_files.Environments.Abstract_Environment.Abstract_Environment cimport Abstract_Environment
from src_files.MyMath.MyMath cimport round_to_int, degree_sin, degree_cos
from src_files.MyMath.cython_debug_helper import cython_debug_call
from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model cimport Normal_model

ctypedef unsigned char map_view_t
cdef class Basic_Car_Environment(Abstract_Environment):
    cdef map_view_t[:, ::1] map_view
    cdef double[::1] rays_degrees
    cdef float[::1] state
    cdef Car car
    cdef int max_steps
    cdef int current_step
    cdef (double, double) start_position
    cdef double start_angle
    cdef double start_speed
    cdef double speed_change
    cdef double rays_distances_scale_factor
    cdef double ray_input_clip

    cdef double angle_max_change
    cdef double collision_reward

    # cdef bint _tmp_safe_rewards

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def p_get_trajectory_results(self, model: Normal_model) -> float:
        """
        Get the results of the trajectory
        :param model:
        :return:
        """
        cdef Normal_model model_c = model
        cdef float[:, ::1] state_tmp = np.empty((1, model_c.normal_input_size), dtype=np.float32)
        cdef double reward = 0.0
        with nogil:
            while self.is_alive():
                state_tmp[0] = self.get_state()
                reward += self.react(model_c.forward_pass(state_tmp)[0])
        return reward

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    def p_get_trajectory_logs(self, model: Normal_model, max_steps: int = -1) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Get the logs of the trajectory
        :param model:
        :param max_steps:
        :return: (states, outputs)
        """
        cdef Normal_model model_c = model
        cdef int max_steps_c = max_steps if max_steps > 0 else self.max_steps
        cdef float[:, ::1] states = np.empty((max_steps_c, model_c.normal_input_size), dtype=np.float32)
        cdef float[:, ::1] outputs = np.empty((max_steps_c, model_c.normal_output_size), dtype=np.float32)
        cdef float[:, ::1] state_tmp = np.empty((1, model_c.normal_input_size), dtype=np.float32)
        cdef float[::1] result_tmp
        cdef double result = 0.0
        cdef int j, row, cols
        row = 0
        cols = model_c.normal_input_size
        with nogil:
            while self.is_alive() and row < max_steps_c:
                state_tmp[0] = self.get_state()
                for j in range(cols):
                    states[row, j] = state_tmp[0, j]
                result_tmp = model_c.forward_pass(state_tmp)[0]
                for j in range(model_c.normal_output_size):
                    outputs[row, j] = result_tmp[j]
                result += self.react(result_tmp)
                row += 1
        return result, np.array(states[:row], copy=False), np.array(outputs[:row], copy=False)

    def p_get_safe_data(self) -> dict[str, Any]:
        return {
            "x": self.car.x,
            "y": self.car.y,
            "angle": self.car.angle,
            "speed": self.car.speed,
            "current_step": self.current_step,
        }

    def p_load_safe_data(self, safe_data: dict[str, Any]) -> None:
        self.car.x = safe_data["x"]
        self.car.y = safe_data["y"]
        self.car.angle = safe_data["angle"]
        self.car.speed = safe_data["speed"]
        self.current_step = safe_data["current_step"]

    def p_react(self, outputs: np.ndarray) -> float:
        """
        React to the outputs of the model
        :param outputs: nparray of the outputs
        :return: reward
        """
        cdef float[::1] outputs_c = outputs
        return self.react(outputs_c)

    def get_car_position(self) -> Tuple[float, float]:
        """
        Get car position
        :return: (x, y)
        """
        return self.car.x, self.car.y

    def get_car_angle(self) -> float:
        """
        Returns car angle in degrees
        :return: angle
        """
        return self.car.angle


    def __init__(self,
                 map_view: np.ndarray = np.zeros((1, 1)),
                 start_position: Tuple[float, float] = (0, 0),
                 start_angle: float = 0,
                 angle_max_change: float = 1,
                 car_dimensions: Tuple[float, float] = (10, 20),
                 initial_speed: float = 0.5,
                 min_speed: float = 0.3,
                 max_speed: float = 1,
                 speed_change: float = 0.05,
                 rays_degrees: Union[List[float], Tuple[float]] = (-90, -45, 0, 45, 90),
                 rays_distances_scale_factor: float = 100,
                 max_steps: int = 1000,
                 ray_input_clip: float = 1000,
                 collision_reward: float = -20,
                 ):
        """

        :param map_view: 2d nparray of the map, 0 is free space, 1 is wall (row, column)
        :param start_position: tuple of the start position (x, y)
        :param start_angle: angle in degrees, right is 0, top is 90
        :param car_dimensions: tuple of the car dimensions (width, height)
        :param speed: speed of the car, currently it is constant
        :param rays_degrees: list of the degrees of the rays, car direction is 0 degrees, so it could be e.g. [-45, -30, -15, 0, 15, 30, 45]
        """

        # if np.random.rand() < 0.001:
        #     self._tmp_safe_rewards = True

        self.map_view = np.array(map_view, dtype=np.uint8)
        self.start_position = start_position
        self.start_angle = start_angle
        self.start_speed = initial_speed
        self.angle_max_change = angle_max_change
        self.rays_distances_scale_factor = rays_distances_scale_factor
        self.max_steps = max_steps
        self.speed_change = speed_change
        self.ray_input_clip = ray_input_clip
        self.current_step = 0
        self.collision_reward = collision_reward

        self.car = Car(
            start_position,
            start_angle,
            car_dimensions,
            initial_speed,
            min_speed,
            max_speed
        )

        self.rays_degrees = np.array(
            [ray for ray in rays_degrees], dtype=np.float64
        )

        # cython_debug_call({
        #     "map_view": np.array(self.map_view),
        #     "np_input_type": real_t_numpy,
        # })
        self.state = np.zeros(len(rays_degrees) + 1, dtype=np.float32)


    cdef int reset(self) noexcept nogil:
        self.current_step = 0
        self.car.set_state(self.start_position[0], self.start_position[1], self.start_angle, self.start_speed)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef float[::1] get_state(self) noexcept nogil:
        cdef double[::1] rays_degrees_here = self.rays_degrees
        cdef float[::1] state_here = self.state

        for i in range(self.rays_degrees.shape[0]):
            state_here[i] = self.get_ray_distance(rays_degrees_here[i]) / self.rays_distances_scale_factor
            if state_here[i] > self.ray_input_clip:
                state_here[i] = self.ray_input_clip

        state_here[state_here.shape[0] - 1] = self.car.speed / self.car.max_speed

        # if self._tmp_safe_rewards:
        #     with gil:
        #         print(list(state_here), end="  ")

        return state_here

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_ray_distance(self, double ray_angle) noexcept nogil:
        cdef map_view_t[:, ::1] map_view_here = self.map_view
        cdef double x = self.car.x
        cdef double y = self.car.y
        cdef double distance = 0
        cdef int check_x, check_y
        cdef double angle = self.car.angle + ray_angle
        cdef double sin_angle = degree_sin(angle)
        cdef double cos_angle = degree_cos(angle)

        x = self.car.x
        y = self.car.y
        check_x = round_to_int(x)
        check_y = round_to_int(y)

        while check_x >= 0 and check_x < map_view_here.shape[1] and check_y >= 0 and check_y < map_view_here.shape[0] and map_view_here[check_y, check_x] == 0:
            x += cos_angle
            y -= sin_angle
            distance += 1
            check_x = round_to_int(x)
            check_y = round_to_int(y)

        return distance


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double react(self, float[::1] outputs) noexcept nogil:
        #self.car.change_angle(self.angle_max_change * outputs[0])
        cdef double result
        cdef int change_index_action = 0
        for i in range(3):
            if outputs[i] > outputs[change_index_action]:
                change_index_action = i
        if change_index_action == 1:
            self.car.change_angle(self.angle_max_change)
        elif change_index_action == 2:
            self.car.change_angle(-self.angle_max_change)


        cdef double engine_power = 0.0
        change_index_action = 3
        for i in range(3, 6):
            if outputs[i] > outputs[change_index_action]:
                change_index_action = i
        if change_index_action == 4:
            engine_power = 1.0
        elif change_index_action == 5:
            engine_power = -1.0

        # cdef double engine_power = outputs[3]

        # TRACTION !!!
        # ADDED !!!
        # engine_power -= self.car.speed / self.car.max_speed
        # END TRACTION !!!
        # cdef double engine_power = 0.0
        # cdef double steering_power = 0.0
        # cdef int max_index = 0
        # cdef int i
        # for i in range(9):
        #     if outputs[i] > outputs[max_index]:
        #         max_index = i
        # if max_index // 3 == 1:
        #     steering_power = 1.0
        # elif max_index // 3 == 2:
        #     steering_power = -1.0
        #
        # if max_index % 3 == 1:
        #     engine_power = 1.0
        # elif max_index % 3 == 2:
        #     engine_power = -1.0
        #
        # self.car.change_angle(self.angle_max_change * steering_power)
        self.car.change_speed(self.speed_change * engine_power)

        self.current_step += 1
        result = self.car.step()
        if self.car.does_collide(self.map_view):
            result += self.collision_reward

        return result

    cdef inline bint is_alive(self) noexcept nogil:
        return self.current_step < self.max_steps and not self.car.does_collide(self.map_view)

    cdef inline int get_state_length(self) noexcept nogil:
        return self.rays_degrees.shape[0] + 1


cdef class Car:
    cdef double x
    cdef double y
    cdef double angle
    cdef double speed
    cdef double min_speed
    cdef double max_speed
    cdef double width
    cdef double height
    cdef double distance_center_corner
    cdef double angle_to_corner

    cdef bint does_collide_memory
    cdef bint is_does_collide_actual

    def __init__(self,
                 start_position: Tuple[float, float],
                 angle: float,
                 car_dimensions: Tuple[float, float],
                 speed: float,
                 min_speed: float,
                 max_speed: float,
                 ):
        """

        :param start_position: tuple of the start position (x, y)
        :param angle: angle in degrees
        :param car_dimensions: tuple of the car dimensions (width, height)
        :param speed: speed of the car, currently it is constant
        """
        self.x = start_position[0]
        self.y = start_position[1]
        self.angle = angle
        self.speed = speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.width = car_dimensions[0]
        self.height = car_dimensions[1]
        self.distance_center_corner = math.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        self.angle_to_corner = math.degrees(math.atan2(self.width / 2, self.height / 2))
        self.is_does_collide_actual = False
        self.does_collide_memory = False

    cdef int set_state(self, double x, double y, double angle, double speed) noexcept nogil:
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.is_does_collide_actual = False
        return 0

    cdef double step(self) noexcept nogil:
        """
        Returns distance of the car
        """
        cdef double return_value
        self.x += self.speed * degree_cos(self.angle)
        self.y -= self.speed * degree_sin(self.angle)
        self.is_does_collide_actual = False


        return_value = (self.speed / self.max_speed) ** 2
        if self.speed <= 0:
            return_value = -1
        return return_value

    cdef int change_speed(self, double speed_change) noexcept nogil:
        """
        Changes speed of the car by the given amount
        """

        self.speed += speed_change
        if self.speed < self.min_speed:
            self.speed = self.min_speed
        elif self.speed > self.max_speed:
            self.speed = self.max_speed
        return 0

    cdef int change_angle(self, double angle) noexcept nogil:
        """
        Changes angle of the car by the given amount
        """
        self.angle += angle
        if self.angle < 0:
            self.angle += 360
        elif self.angle >= 360:
            self.angle -= 360
        return 0

    cdef bint does_collide(self, map_view_t[:, ::1] map_view) noexcept nogil:
        if not self.is_does_collide_actual:
            self.does_collide_memory = (self.does_collide_one(map_view, self.distance_center_corner, self.angle - self.angle_to_corner) or
                                        self.does_collide_one(map_view, self.distance_center_corner, self.angle + self.angle_to_corner) or
                                        self.does_collide_one(map_view, self.distance_center_corner, 180 + self.angle - self.angle_to_corner) or
                                        self.does_collide_one(map_view, self.distance_center_corner, 180 + self.angle + self.angle_to_corner) or
                                        self.does_collide_one(map_view, self.width / 2, self.angle + 90) or
                                        self.does_collide_one(map_view, self.width / 2, self.angle - 90) or
                                        self.does_collide_one(map_view, self.height / 2, self.angle) or
                                        self.does_collide_one(map_view, self.height / 2, self.angle + 180))
            self.is_does_collide_actual = True
        return self.does_collide_memory

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint does_collide_one(self, map_view_t[:, ::1] map_view, double distance, double angle) noexcept nogil:
        cdef int check_x, check_y
        check_x = round_to_int(self.x + distance * degree_cos(angle))
        check_y = round_to_int(self.y - distance * degree_sin(angle))

        if check_x < 0 or check_x >= map_view.shape[1]:
            return True
        if check_y < 0 or check_y >= map_view.shape[0]:
            return True

        return map_view[check_y, check_x] == 1
