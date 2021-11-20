import json

from ROAR.control_module.controller import Controller
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque
import numpy as np
from ROAR_iOS.config_model import iOSConfig
from pathlib import Path


class RealWorldImageBasedPIDController(Controller):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        long_error_deque_length = 10
        lat_error_deque_length = 10
        self.lat_error_queue = deque(maxlen=lat_error_deque_length)  # this is how much error you want to accumulate
        self.long_error_queue = deque(maxlen=long_error_deque_length)  # this is how much error you want to accumulate
        self.target_speed = 10  # m / s
        self.config = json.load(Path(self.agent.agent_settings.pid_config_file_path).open('r'))
        self.long_config = self.config["longitudinal_controller"]
        self.lat_config = self.config["latitudinal_controller"]

    def run_in_series(self, next_waypoint=None, **kwargs) -> VehicleControl:
        steering = self.lateral_pid_control()
        throttle = self.long_pid_control()
        return VehicleControl(throttle=throttle, steering=steering)

    def lateral_pid_control(self) -> float:
        error = self.agent.kwargs.get("lat_error", 0)
        error_dt = 0 if len(self.lat_error_queue) == 0 else error - self.lat_error_queue[-1]
        self.lat_error_queue.append(error)
        error_it = sum(self.lat_error_queue)
        k_p, k_d, k_i = self.find_k_values(self.agent.vehicle, self.lat_config)
        # print(f"Speed = {self.agent.vehicle.get_speed(self.agent.vehicle)}"
        #       f"kp, kd, ki = {k_p, k_d, k_i} ")

        e_p = k_p * error
        e_d = k_d * error_dt
        e_i = k_i * error_it
        lat_control = np.clip((e_p + e_d + e_i), -1, 1)
        # print(f"speed = {self.agent.vehicle.get_speed(self.agent.vehicle)} "
        #       f"e = {round((e_p + e_d + e_i), 3)}, "
        #       f"e_p={round(e_p, 3)},"
        #       f"e_d={round(e_d, 3)},"
        #       f"e_i={round(e_i, 3)},"
        #       f"lat_control={lat_control}")
        # print()
        return lat_control

    def long_pid_control(self) -> float:
        k_p, k_d, k_i = self.find_k_values(self.agent.vehicle, self.long_config)
        e = self.target_speed - self.agent.vehicle.get_speed(self.agent.vehicle)
        neutral = -90
        incline = self.agent.vehicle.transform.rotation.pitch - neutral
        e = e * - 1 if incline < -10 else e
        self.long_error_queue.append(e)
        de = 0 if len(self.long_error_queue) < 2 else self.long_error_queue[-2] - self.long_error_queue[-1]
        ie = 0 if len(self.long_error_queue) < 2 else np.sum(self.long_error_queue)
        incline = np.clip(incline, -20, 20)

        e_p = k_p * e
        e_d = k_d * de
        e_i = k_i * ie
        e_incline = 0.015 * incline
        total_error = e_p + e_d + e_i + e_incline
        long_control = np.clip(total_error, 0, 1)
        # print(f"speed = {self.agent.vehicle.get_speed(self.agent.vehicle)} "
        #       f"e = {round(total_error,3)}, "
        #       f"e_p={round(e_p,3)},"
        #       f"e_d={round(e_d,3)},"
        #       f"e_i={round(e_i,3)},"
        #       f"e_incline={round(e_incline, 3)}, "
        #       f"long_control={long_control}")
        return long_control

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])
