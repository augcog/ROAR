from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np
from ROAR_iOS.config_model import iOSConfig
from pathlib import Path

class SimplePIDController(Controller):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.lat_error_queue = deque(maxlen=10)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=10)  # this is how much error you want to accumulate

        self.target_speed = 3  # m / s
        ios_config_file_path = Path("ROAR_iOS/configurations/ios_config.json")
        self.ios_config: iOSConfig = iOSConfig.parse_file(ios_config_file_path)

        self.lat_kp = 0.006  # this is how much you want to steer
        self.lat_kd = 0.075  # this is how much you want to resist change
        self.lat_ki = 0.00025  # this is the correction on past error

        self.uphill_long_pid = {
            "long_kp": 0.25,
            "long_kd": 0.2,
            "long_ki": 0.1,
        }
        self.flat_long_pid = {
            "long_kp": 0.13,
            "long_kd": 0.15,
            "long_ki": 0.05,
        }
        self.downhill_long_pid = {
            "long_kp": 0.15,
            "long_kd": 0.05,
            "long_ki": 0
        }

    def run_in_series(self, next_waypoint=None, **kwargs) -> VehicleControl:
        steering = self.lateral_pid_control()
        throttle = self.long_pid_control()
        return VehicleControl(throttle=throttle, steering=steering)

    def lateral_pid_control(self) -> float:
        error = self.agent.kwargs.get("lat_error", 0)
        error_dt = 0 if len(self.lat_error_queue) == 0 else error - self.lat_error_queue[-1]
        self.lat_error_queue.append(error)
        error_it = sum(self.lat_error_queue)

        e_p = self.lat_kp * error
        e_d = self.lat_kd * error_dt
        e_i = self.lat_ki * error_it
        lat_control = np.clip((e_p + e_d + e_i), -1, 1)
        return lat_control

    def long_pid_control(self) -> float:
        kp = 1
        kd = 0
        ki = 0

        e = self.target_speed - self.agent.vehicle.get_speed(self.agent.vehicle)
        neutral = -90
        incline = self.agent.vehicle.transform.rotation.pitch - neutral
        e = e * - 1 if incline < 10 else e
        self.long_error_queue.append(e)
        de = 0 if len(self.long_error_queue) < 2 else self.long_error_queue[-2] - self.long_error_queue[-1]
        ie = 0 if len(self.long_error_queue) < 2 else np.sum(self.long_error_queue)
        incline = np.clip(incline, -20, 20)

        e_p = kp * e
        e_d = kd * de
        e_i = ki * ie
        e_incline = 0.015 * incline
        long_control = np.clip(e_p+e_d+e_i+e_incline, -self.ios_config.max_throttle, self.ios_config.max_throttle)
        print(f"e_p={e_p},e_d={e_d},e_i={e_i},e_incline={e_incline}, long_control={long_control}")
        return long_control


        # error = self.target_speed - curr_speed
        # error = error * -1 if incline < -10 else error
        # if incline > 10:
        #     # up hill
        #     kp, kd, ki = self.uphill_long_pid["long_kp"], self.uphill_long_pid["long_kd"], self.uphill_long_pid[
        #         "long_ki"]
        # elif incline < -10:
        #     # downhill
        #     kp, kd, ki = self.downhill_long_pid["long_kp"], self.downhill_long_pid["long_kd"], \
        #                  self.downhill_long_pid["long_ki"]
        # else:
        #     kp, kd, ki = self.flat_long_pid["long_kp"], self.flat_long_pid["long_kd"], self.flat_long_pid["long_ki"]

        # error_dt = 0 if len(self.long_error_queue) == 0 else error - self.long_error_queue[-1]
        # self.long_error_queue.append(error)
        # error_it = sum(self.long_error_queue)
        # e_p = kp * error
        # e_d = kd * error_dt
        # e_i = ki * error_it
        # long_control = np.clip(e_p + e_d + e_i, -self.ios_config.max_throttle, self.ios_config.max_throttle)
        # print(long_control, error, e_p, e_d, e_i)
        # return long_control
