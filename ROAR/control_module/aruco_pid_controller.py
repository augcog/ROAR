from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np


class SimplePIDController(Controller):
    def __init__(self, agent, distance_to_keep=0.5, center_x=-0.2, **kwargs):
        super().__init__(agent, **kwargs)

        self.yaw_error_buffer = deque(maxlen=20)

        self.lat_error_queue = deque(maxlen=20)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=50)  # this is how much error you want to accumulate
        self.center_x = center_x
        self.lat_kp = 1  # this is how much you want to steer
        self.lat_kd = 0  # this is how much you want to resist change
        self.lat_ki = 0.005  # this is the correction on past error
        self.x_error_weight = 1
        self.yaw_error_weight = 0.9

        self.distance_to_keep = distance_to_keep
        self.max_throttle = 0.17
        self.lon_kp = 0.16  # this is how much you want to steer
        self.lon_kd = 0.1  # this is how much you want to resist change
        self.lon_ki = 0.02  # this is the correction on past error

    def run_in_series(self, next_waypoint=None, **kwargs) -> VehicleControl:
        control = VehicleControl()
        self.lateral_pid_control(next_waypoint=next_waypoint, control=control)
        self.long_pid_control(next_waypoint=next_waypoint, control=control)
        print(control)
        return control

    def lateral_pid_control(self, next_waypoint: Transform, control: VehicleControl):
        x_error = (next_waypoint.location.x - self.center_x) / next_waypoint.location.z
        yaw_error = np.deg2rad(next_waypoint.rotation.yaw - 0)
        # there are error in yaw detection, i am resolving the error by using an average.
        self.yaw_error_buffer.append(yaw_error)
        error = x_error * self.x_error_weight + np.average(self.yaw_error_buffer) * self.yaw_error_weight
        error_dt = 0 if len(self.lat_error_queue) == 0 else error - self.lat_error_queue[-1]
        self.lat_error_queue.append(error)
        error_it = sum(self.lat_error_queue)

        e_p = self.lat_kp * error
        e_d = self.lat_kd * error_dt
        e_i = self.lat_ki * error_it
        lat_control = np.clip((e_p + e_d + e_i), -1, 1)
        control.steering = lat_control

    def long_pid_control(self, next_waypoint: Transform, control: VehicleControl):
        dist_to_car = next_waypoint.location.z
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
