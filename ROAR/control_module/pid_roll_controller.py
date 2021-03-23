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

from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.color_plane_detector import ColorPlaneDetector

class PIDController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
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
        # print(self.agent.vehicle.transform.rotation.roll)
        # print('steering', steering)
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
        self.gpd = GroundPlaneDetector(agent, vis=True)
        self.cpd = ColorPlaneDetector(agent)
        self._dt = dt
        self.time_step = 0

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed))
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config=self.config)
        error = target_speed - current_speed

        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            # print(self._error_buffer[-1], self._error_buffer[-2])
            _de = abs((self._error_buffer[-2] - self._error_buffer[-1])) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        output = float(np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.throttle_boundary[0],
                               self.throttle_boundary[1]))

        if self.time_step % 5 == 0:
            # self.gpd.run_in_series()
            self.gpd.run_in_series()
            road_normal = np.array(self.gpd.road_normal)
            road_dot = np.dot(road_normal, np.array([0, 1, 0]))
            true_roll = self.agent.vehicle.transform.rotation.roll
            pred_roll = np.arccos(road_dot)

            # print(road_normal)
            # print(np.arccos(road_dot))
            # print(self.agent.vehicle.transform.rotation.roll)
            # print(np.abs(true_roll - pred_roll))

            self.time_step = self.time_step % 5
        self.time_step += 1

        use_plane_roll = True
        pred_roll = self.agent.vehicle.transform.rotation.roll
        if use_plane_roll:
            pred_roll = self.gpd.pred_roll

        output = np.exp(-0.048 * np.abs(pred_roll))

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
        #self.steering_boundary = (-.1, .1)

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
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()

        print(v_begin)
        print('next wp x: ', next_waypoint.location.x)
        print('next wp z: ', next_waypoint.location.z)
        print('next wp y: ', next_waypoint.location.y)

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

        k_p, k_d, k_i = LongPIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        # print(f"v_vec_normed: {v_vec_normed} | w_vec_normed = {w_vec_normed}")
        # print("v_vec_normed @ w_vec_normed.T:", v_vec_normed @ w_vec_normed.T)
        # print(f"Curr: {self.agent.vehicle.transform.location}, waypoint: {next_waypoint}")
        # print(f"lat_control: {round(lat_control, 3)} | error: {error} ")
        # print()
        return lat_control

    # def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
    #     # calculate a vector that represent where you are going
    #     v_begin = self.agent.vehicle.transform.location
    #     v_end = v_begin + Location(
    #         x=math.cos(math.radians(self.agent.vehicle.transform.rotation.yaw)),
    #         y=v_begin.y,
    #         z=math.sin(math.radians(self.agent.vehicle.transform.rotation.yaw)),
    #     )
    #     v_vec = np.array([v_end.x - v_begin.x,v_end.y - v_begin.y, v_end.z - v_begin.z])
    #
    #     # calculate error projection
    #     w_vec = np.array(
    #         [
    #             next_waypoint.location.x - v_begin.x,
    #             next_waypoint.location.y - v_begin.y,
    #             next_waypoint.location.z - v_begin.z,
    #         ]
    #     )
    #     _dot = math.acos(
    #         np.clip(
    #             np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
    #             -1.0,
    #             1.0,
    #         )
    #     )
    #     _cross = np.cross(v_vec, w_vec)
    #     if _cross[1] > 0:
    #         _dot *= -1.0
    #     self._error_buffer.append(_dot)
    #     if len(self._error_buffer) >= 2:
    #         _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
    #         _ie = sum(self._error_buffer) * self._dt
    #     else:
    #         _de = 0.0
    #         _ie = 0.0
    #
    #     # print('_dot PIDcontroller = ', _dot)
    #     k_p, k_d, k_i = LongPIDController.find_k_values(
    #         config=self.config, vehicle=self.agent.vehicle)
    #
    #     lat_control = float(
    #         np.clip((k_p * _dot) + (k_d * _de) + (k_i * _ie),
    #         self.steering_boundary[0], self.steering_boundary[1])
    #     )
    #     # print('lat_control = steering?', lat_control)
    #     return lat_control