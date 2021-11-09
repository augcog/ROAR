from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.utilities_module.data_structures_models import Transform, Location
from collections import deque
import numpy as np
import math
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple, Any
from stable_baselines import DDPG
from pathlib import Path


class PIDController(Controller):
    def __init__(self, agent: Agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.long_pid_controller = LongPIDController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed)
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)
        try:
            self.pid_rl_model = DDPG.load(Path("./ROAR_Sim/data/weights/rl_pid_model.zip"))
        except:
            path = Path(self.agent.kwargs['kwargs']["rl_pid_model_file_path"])
            self.pid_rl_model = DDPG.load(load_path=path)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        action, _ = self.pid_rl_model.predict(self._get_obs())
        target_speed, long_k_p, long_k_d, long_k_i, lat_k_p, lat_k_d, lat_k_i = self.get_k_values(action=action)
        kwargs["target_speed"], kwargs["long_k_p"], kwargs["long_k_d"], kwargs["long_k_i"], kwargs["lat_k_p"], \
        kwargs["lat_k_d"], kwargs["lat_k_i"] = target_speed, long_k_p, long_k_d, long_k_i, lat_k_p, lat_k_d, lat_k_i
        # print(kwargs)
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint, kwargs=kwargs)
        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint, kwargs=kwargs)
        # print(target_speed, steering)
        return VehicleControl(throttle=throttle, steering=steering)

    def _get_obs(self) -> Any:
        curr_speed = np.array([Vehicle.get_speed(self.agent.vehicle),
                               self.agent.vehicle.control.throttle,
                               self.agent.vehicle.control.steering])
        return curr_speed

    def get_k_values(self, action):
        target_speed = action[0]
        long_k_p, long_k_d, long_k_i = action[1], action[2], action[3]
        lat_k_p, lat_k_d, lat_k_i = action[4], action[5], action[6]
        return target_speed, long_k_p, long_k_d, long_k_i, lat_k_p, lat_k_d, lat_k_i


class LongPIDController(Controller):
    def __init__(self, agent, throttle_boundary: Tuple[float, float], max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        target_speed = min(self.max_speed, kwargs.get("kwargs").get("target_speed", self.max_speed))
        # self.logger.debug(f"Target_Speed: {target_speed} | max_speed = {self.max_speed}")
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        k_p, k_d, k_i = kwargs.get("long_k_p", 1), kwargs.get("long_k_d", 0), kwargs.get("long_k_i", 0)
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
        return output

    def find_k_values(self) -> Tuple[float, float, float]:
        k_p = self.agent.kwargs.get("long_k_p", 1)
        k_d = self.agent.kwargs.get("long_k_d", 0)
        k_i = self.agent.kwargs.get("long_k_i", 0)
        return k_p, k_d, k_i


class LatPIDController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
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

        k_p, k_d, k_i = kwargs.get("kwargs").get("lat_k_p", 1), kwargs.get("kwargs").get("lat_k_d", 0), \
                        kwargs.get("kwargs").get("lat_k_i", 0)

        lat_control = float(
            np.clip((k_p * _dot) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        return lat_control
