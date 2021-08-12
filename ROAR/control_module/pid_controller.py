from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import Transform, Location
from collections import deque
import numpy as np
import math
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple
import json
from pathlib import Path


class PIDController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.target_speed = self.agent.agent_settings.target_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        if self.agent.agent_settings.global_pid_values: # ROAR Academy
            self.global_pid_config = self.agent.agent_settings.global_pid_values # ROAR Academy
        else:
            self.global_pid_config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.regional_pid_config = self.agent.agent_settings.regional_pid_values # ROAR Academy
        self.long_pid_controller = LongPIDController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed,
                                                     global_config=self.global_pid_config["longitudinal_controller"],
                                                     regional_config=self.regional_pid_config
                                                     )
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            global_config=self.global_pid_config["latitudinal_controller"],
            regional_config=self.regional_pid_config,
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.target_speed)) # ROAR-Academy: regional target_speed to be updated
        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint)
        return VehicleControl(throttle=throttle, steering=steering)

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


class LongPIDController(Controller):
    def __init__(self, agent, global_config: dict, regional_config: dict, throttle_boundary: Tuple[float, float], 
                max_speed: float,
                dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.global_pid_config = global_config
        self.regional_pid_config = regional_config
        self.max_speed = max_speed
        self.target_speed = self.agent.agent_settings.target_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)

        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        # print("long_next_waypoints:", next_waypoint)
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.target_speed))
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        
        current_region = self.agent.local_planner.current_region
        if current_region in self.regional_pid_config:
            pid_config = self.regional_pid_config[current_region]["longitudinal_controller"]
        else:
            pid_config = self.global_pid_config 

        k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config = pid_config)
        # print("long pid values:", k_p, k_d, k_i) # ROAR_Academy
        error = target_speed - current_speed

        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            # print(self._error_buffer[-1], self._error_buffer[-2])
            _de = (self._error_buffer[-2] - self._error_buffer[-1]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        output = float(np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.throttle_boundary[0],
                               self.throttle_boundary[1]))
        # self.logger.debug(f"curr_speed: {round(current_speed, 2)} | kp: {round(k_p, 2)} | kd: {k_d} | ki = {k_i} | "
        #       f"err = {round(error, 2)} | de = {round(_de, 2)} | ie = {round(_ie, 2)}")
              # f"self._error_buffer[-1] {self._error_buffer[-1]} | self._error_buffer[-2] = {self._error_buffer[-2]}")
        return output


class LatPIDController(Controller):
    def __init__(self, agent, global_config: dict, regional_config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.global_pid_config = global_config
        self.regional_pid_config = regional_config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # print("lat_next_waypoints:", next_waypoint)
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
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        current_region = self.agent.local_planner.current_region # ROAR-Academy
        if current_region in self.regional_pid_config:
            pid_config = self.regional_pid_config[current_region]["latitudinal_controller"]
        else:
            pid_config = self.global_pid_config 
        k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config=pid_config)
        # print("lat pid values:", k_p, k_d, k_i) # ROAR_Academy
        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        return lat_control
