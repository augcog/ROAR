from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np
import math


class UDP_PID_CONTROLLER(Controller):
    def __init__(self, agent, distance_to_keep=0.5, center_x=-0.2, **kwargs):
        super().__init__(agent, **kwargs)

        self.lat_error_queue = deque(maxlen=20)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=50)  # this is how much error you want to accumulate
        self.center_x = center_x
        self.lat_kp = 1  # this is how much you want to steer
        self.lat_kd = 0  # this is how much you want to resist change
        self.lat_ki = 0.005  # this is the correction on past error
        self.dt = 0.03

        self.distance_to_keep = distance_to_keep
        self.max_throttle = 0.18
        self.lon_kp = 0.17  # this is how much you want to steer
        self.lon_kd = 0.1  # this is how much you want to resist change
        self.lon_ki = 0.025  # this is the correction on past error

    def run_in_series(self, target_point=None, **kwargs) -> VehicleControl:
        control = VehicleControl()
        self.lateral_pid_control(target_point, control=control)
        self.long_pid_control(target_point, control=control)
        return control

    def lateral_pid_control(self, target_point, control: VehicleControl):
        # calculate a vector that represent where you are going
        self_point = self.agent.vehicle.transform.to_array()
        v_begin = self_point[:3]
        direction_vector = np.array([
            -np.sin(np.deg2rad(self_point[5])),
            0,
            -np.cos(np.deg2rad(self_point[5]))
        ])
        # calculate error projection
        w_vec = np.array([
            target_point[1] - v_begin[0],
            0,
            target_point[2] - v_begin[2]
        ])
        v_vec_normed = direction_vector / np.linalg.norm(direction_vector)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self.lat_error_queue.append(error)
        if len(self.lat_error_queue) >= 2:
            error_dt = (self.lat_error_queue[-1] - self.lat_error_queue[-2]) / self.dt
            error_it = sum(self.lat_error_queue) * self.dt
        else:
            error_dt = 0.0
            error_it = 0.0

        e_p = self.lat_kp * error
        e_d = self.lat_kd * error_dt
        e_i = self.lat_ki * error_it
        raw_steering = e_p + e_d + e_i
        lat_control = np.clip(raw_steering, -1, 1)
        control.steering = lat_control

    def long_pid_control(self, target_point, control: VehicleControl):
        self_point = self.agent.vehicle.transform.to_array()

        x_diff = target_point[0] - self_point[0]
        y_diff = target_point[1] - self_point[1]
        z_diff = target_point[2] - self_point[2]
        dist_to_car = math.sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff)
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
