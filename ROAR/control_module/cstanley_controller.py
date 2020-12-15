
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


class CStanley_controller(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.long_pid_controller = LongPIDController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed,
                                                     config=self.config["longitudinal_controller"])
        self.clat_stanley_controller = CLatStanley_controller(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
        steering = self.clat_stanley_controller.run_in_series(next_waypoint=next_waypoint)


        print('steering', steering)
        print('--------------------')


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
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        k_p, k_d, k_i = CStanley_controller.find_k_values(vehicle=self.agent.vehicle, config=self.config)
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


class CLatStanley_controller(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform,
                      **kwargs) -> float:  # *********** aka stanley_control(state, cx, cy, cyaw, last_target_idx)
        '''
        *** inputs needed: vehicle yaw, x, y; nearest path yaw, x, y
        *** implement target calculations: heading error (vehicle yaw - path yaw at nearest path point);
                                           cross track error (front axle x,y - nearest path x,y)
        self.way_points_queue[0]
        *** output lat_control:  steering angle delta = heading error + inv tan (gain * cross track error/veh speed)
        '''

        return self.stan_calcs(next_waypoint=next_waypoint)

    def stan_calcs(self, next_waypoint: Transform, **kwargs) -> float:
        '''
        calculate target
        front axle position (corrected from veh location + lv cos and sin heading
        nearest point
        front axle error (distance from front axle to desired path position)
          front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
          error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
        '''
        controls_gain = k = 0.5

        target_pitch = next_waypoint.rotation.pitch
        curr_pitch = self.agent.vehicle.transform.rotation.pitch

        curr_velocity = Vehicle.get_speed(self.agent.vehicle)

        # # Project RMS error onto front axle vector
        # front_axle_vec = [-np.cos(curr_pitch + np.pi / 2),
        #                   -np.sin(curr_pitch + np.pi / 2)]
        error_front_axle = self.calc_cross_track_error(next_waypoint) #np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        # theta_e corrects the heading error
        theta_e = np.deg2rad(target_pitch) - np.deg2rad(curr_pitch)
        # theta_d corrects the cross track error
        theta_d = k*error_front_axle

        # Steering control
        delta = theta_e + theta_d

        print('-------------------')
        print('target pitch = ', target_pitch)
        print('vehicle pitch = ', curr_pitch)
        print('heading error', theta_d)
        print('cross track error', theta_e)
        print('delta', delta)
        # print(f"target_pitch = {target_pitch} | curr_pitch = {curr_pitch} | theta_e = {theta_e} | theta_d = {theta_d} | delta = {delta}")
        return delta

    def calc_curr_heading_err(self, next_waypoint: Transform) -> float:
        return next_waypoint.rotation.pitch - self.agent.vehicle.transform.rotation.pitch

    def calc_cross_track_error(self, next_waypoint:Transform) -> float:
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(self.agent.vehicle.transform.rotation.pitch)),
            y=v_begin.y,
            z=math.sin(math.radians(self.agent.vehicle.transform.rotation.pitch)),
        )
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, v_end.z - v_begin.z])

        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin.x,
                next_waypoint.location.y - v_begin.y,
                next_waypoint.location.z - v_begin.z,
            ]
        )
        _dot = math.acos(
            np.clip(
                np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
                -1.0,
                1.0,
            )
        )
        _cross = np.cross(v_vec, w_vec)
        if _cross[1] > 0:
            _dot *= -1.0
        return _dot