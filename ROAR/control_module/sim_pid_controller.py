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
        self.max_speed = self.agent.agent_config.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_config.pid_config_file_path).open(mode='r'))
        self.long_pid_controller = LongPIDController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed, config=self.config["longitudinal_controller"])
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint)

        if abs(steering) > 0.3 and Vehicle.get_speed(self.agent.vehicle) > 50:
            print("TOO FAST, FORCED THROTTLE DECREASE")
            throttle = -1
        if abs(steering) < 0.1:
            throttle = 1
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
        return np.clip([k_p, k_d, k_i], a_min=0, a_max=1)


class LongPIDController(Controller):
    def __init__(self, agent, config: dict, throttle_boundary: Tuple[float, float], max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)

        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed))
        # self.logger.debug(f"Target_Speed: {target_speed} | max_speed = {self.max_speed}")
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config=self.config)
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
              #f"self._error_buffer[-1] {self._error_buffer[-1]} | self._error_buffer[-2] = {self._error_buffer[-2]}")
        return output


class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.control.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(self.agent.vehicle.control.rotation.pitch)),
            y=0,
            z=math.sin(math.radians(self.agent.vehicle.control.rotation.pitch)),
        )
        v_vec = np.array([v_end.x - v_begin.x, 0, v_end.z - v_begin.z])
        v_vec = np.array([v_end.x - v_begin.x, 0, v_end.z - v_begin.z])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin.x,
                0,
                next_waypoint.location.z - v_begin.z,
            ]
        )
        _dot = math.acos(
            np.clip(
                np.dot(v_vec, w_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
                -1.0,
                1.0,
            )
        )
        _cross = np.cross(v_vec, w_vec)
        if _cross[1] > 0:
            _dot *= -1
        self._error_buffer.append(_dot)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = PIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * _dot) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        return lat_control