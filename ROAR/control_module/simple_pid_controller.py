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
        long_error_deque_length = 10
        lat_error_deque_length = 10
        self.lat_error_queue = deque(maxlen=lat_error_deque_length)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=long_error_deque_length)  # this is how much error you want to accumulate

        self.target_speed = 5  # m / s
        ios_config_file_path = Path("ROAR_iOS/configurations/ios_config.json")
        self.ios_config: iOSConfig = iOSConfig.parse_file(ios_config_file_path)

        self.lat_kp = 0.006  # this is how much you want to steer
        self.lat_kd = 0.075  # this is how much you want to resist change
        self.lat_ki = 0.00025  # this is the correction on past error

        self.uphill_long_pid = {
            "long_kp": 0.25,
            "long_kd": 0.2,
            "long_ki": 0.05 / long_error_deque_length,
        }
        self.flat_long_pid = {
            "long_kp": 0.1,
            "long_kd": 0.0,
            "long_ki": 0.05 / long_error_deque_length,
        }
        self.downhill_long_pid = {
            "long_kp": 0.15,
            "long_kd": 0.05,
            "long_ki": 0 / long_error_deque_length
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
        kp = self.flat_long_pid.get("long_kp", 1)
        kd = self.flat_long_pid.get("long_kd", 0)
        ki = self.flat_long_pid.get("long_ki", 0)

        e = self.target_speed - self.agent.vehicle.get_speed(self.agent.vehicle)
        neutral = -90
        incline = self.agent.vehicle.transform.rotation.pitch - neutral
        e = e * - 1 if incline < -10 else e
        self.long_error_queue.append(e)
        de = 0 if len(self.long_error_queue) < 2 else self.long_error_queue[-2] - self.long_error_queue[-1]
        ie = 0 if len(self.long_error_queue) < 2 else np.sum(self.long_error_queue)
        incline = np.clip(incline, -20, 20)

        e_p = kp * e
        e_d = kd * de
        e_i = ki * ie
        e_incline = 0.015 * incline
        total_error = e_p+e_d+e_i+e_incline
        long_control = np.clip(total_error, 0, self.ios_config.max_throttle)
        print(f"e = {round(total_error,3)}, "
              f"e_p={round(e_p,3)},"
              f"e_d={round(e_d,3)},"
              f"e_i={round(e_i,3)},"
              f"e_incline={round(e_incline, 3)}, "
              f"long_control={long_control}")
        return long_control
