from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np
import math
from typing import List


class UDP_PID_CONTROLLER(Controller):
    def __init__(self, agent, distance_to_keep=0.5, center_x=-0.2, **kwargs):
        super().__init__(agent, **kwargs)

        self.lat_error_queue = deque(maxlen=20)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=50)  # this is how much error you want to accumulate
        self.center_x = center_x
        self.lat_kp = 1  # this is how much you want to steer
        self.lat_kd = 0  # this is how much you want to resist change
        self.lat_ki = 0.005  # this is the correction on past error
        self._dt = 0.03

        self.distance_to_keep = distance_to_keep
        self.max_throttle = 0.18
        self.lon_kp = 0.17  # this is how much you want to steer
        self.lon_kd = 0.1  # this is how much you want to resist change
        self.lon_ki = 0.025  # this is the correction on past error

    def run_in_series(self, target_point: List, **kwargs) -> VehicleControl:
        next_waypoint: Transform = Transform.from_array(target_point)
        control = VehicleControl()
        self.lateral_pid_control(next_waypoint, control=control)
        self.long_pid_control(target_point, control=control)
        return control

    def lateral_pid_control(self, next_waypoint: Transform, control: VehicleControl):
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self.lat_error_queue.append(error)
        if len(self.lat_error_queue) >= 2:
            _de = (self.lat_error_queue[-1] - self.lat_error_queue[-2]) / self._dt
            _ie = sum(self.lat_error_queue) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        # k_p, k_d, k_i = PIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)
        k_p, k_d, k_i = 1, 0, 0

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), -1, 1)
        )
        control.steering = lat_control
        print(control, v_vec_normed, w_vec_normed, next_waypoint.location, v_begin)

    def long_pid_control(self, target_point, control: VehicleControl):
        self_point = self.agent.vehicle.transform.to_array()

        x_diff = target_point[0] - self_point[0]
        y_diff = target_point[1] - self_point[1]
        z_diff = target_point[2] - self_point[2]
        dist_to_car = math.sqrt(x_diff * x_diff + z_diff * z_diff)
        if dist_to_car < self.distance_to_keep:
            self.logger.info("TOO CLOSE BRAKING!")
            control.brake = True
            control.throttle = 0
        else:
            error = dist_to_car - self.distance_to_keep
            error_dt = 0 if len(self.long_error_queue) == 0 else error - self.long_error_queue[-1]
            self.long_error_queue.append(error)
            error_it = sum(self.long_error_queue)
            e_p = self.lon_kp * error
            e_d = self.lon_kd * error_dt
            e_i = self.lon_ki * error_it
            long_control = np.clip(e_p + e_d + e_i, -1, self.max_throttle)
            control.throttle = long_control
