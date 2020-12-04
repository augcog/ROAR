
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


class Stanley_controller(Controller):
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
        self.lat_pid_controller = LatStanley_controller(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
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

        k_p, k_d, k_i = Stanley_controller.find_k_values(vehicle=self.agent.vehicle, config=self.config)
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

class LatStanley_controller(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt


    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float: #*********** aka stanley_control(state, cx, cy, cyaw, last_target_idx)
        '''
        TODO:  get nearest path yaw; confirm axis car and world;

        *** inputs needed: vehicle yaw, x, y; nearest path yaw, x, y

        *** implement target calculations: heading error (vehicle yaw - path yaw at nearest path point);
                                           cross track error (front axle x,y - nearest path x,y)

        self.way_points_queue[0]

        *** output lat_control:  steering angle delta = heading error + inv tan (gain * cross track error/veh speed)
        '''

        vel = self.agent.vehicle.velocity
        # veh_spd = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) #*** m/s


        k = 0.5 #control gain

        # veh_loc = self.agent.vehicle.transform.location
        veh_x = self.agent.vehicle.transform.location.x
        veh_y = self.agent.vehicle.transform.location.y
        veh_yaw = self.agent.vehicle.transform.rotation.yaw  #***** is this in radians or angle?  guessing angle

        pos_err, head_err = self.stan_calcs(veh_x, veh_y, veh_yaw)

        #lat_control = head_err + k * pos_err #if angle > 30 then 1, otherwise angle/180 ************ what does 1 equate to?  30 degrees?

        lat_control = float(
                    np.clip((head_err + k * pos_err)/30, self.steering_boundary[0], self.steering_boundary[1])   #**** guessing steering of '1' equates to 30 degrees
                )

        return lat_control


    def stan_calcs(self, veh_x, veh_y, veh_yaw, next_waypoint: Transform, **kwargs):

        '''
        calculate target

        front axle position (corrected from veh location + lv cos and sin heading
        nearest point

        front axle error (distance from front axle to desired path position)

          front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
          error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
        '''

        wb = 2.96  # assumed vehicle wheelbase (tesla)

        frontx = veh_x + wb*np.cos(veh_yaw)/2
        fronty = veh_y + wb*np.sin(veh_yaw)/2

        path_x = next_waypoint.location.x  #*** next waypoint: self.way_points_queue[0]
        path_y = next_waypoint.location.y  #** how get

        next_pathpoint = self.way_points_queue[1]
        nx = next_pathpoint.x
        ny = next_pathpoint.y


        #*** calculate distance to next waypoint ***
        # *** calculate front axle position error from path

        dx = [frontx - nx]
        dy = [fronty - ny]
        dpath = np.hypot(dx, dy)


        # front_axle_vec = [-np.cos(veh_yaw + np.pi / 2), -np.sin(veh_yaw + np.pi / 2)]   # RMS error?
        # e_front_axle_pos = np.dot([nx, ny], front_axle_vec)


        path_yaw = np.arctan((ny - path_y) / (nx - path_x))
        #path_yaw = math.atan2((ny - path_y) / (nx - path_x))

        head_err = path_yaw - veh_yaw

        return dpath, head_err

        '''
        *** end my code ***
        '''


        # # calculate a vector that represent where you are going
        # v_begin = self.agent.vehicle.transform.location
        # v_end = v_begin + Location(
        #     x=math.cos(math.radians(self.agent.vehicle.transform.rotation.pitch)),
        #     y=v_begin.y,
        #     z=math.sin(math.radians(self.agent.vehicle.transform.rotation.pitch)),
        # )
        # v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, v_end.z - v_begin.z])
        #
        # ksc = 0.5
        # kpsc =1.0
        #
        #
        #
        # # calculate error projection
        # w_vec = np.array(
        #     [
        #         next_waypoint.location.x - v_begin.x,
        #         next_waypoint.location.y - v_begin.y,
        #         next_waypoint.location.z - v_begin.z,
        #     ]
        # )
        # _dot = math.acos(
        #     np.clip(
        #         np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
        #         -1.0,
        #         1.0,
        #     )
        # )
        # _cross = np.cross(v_vec, w_vec)
        # if _cross[1] > 0:
        #     _dot *= -1.0
        # self._error_buffer.append(_dot)
        # if len(self._error_buffer) >= 2:
        #     _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
        #     _ie = sum(self._error_buffer) * self._dt
        # else:
        #     _de = 0.0
        #     _ie = 0.0
        #
        # k_p, k_d, k_i = Stanley_controller.find_k_values(config=self.config, vehicle=self.agent.vehicle)
        #
        # lat_control = float(
        #     np.clip((k_p * _dot) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        # )
        # return lat_control

#
# class LatPIDController(Controller):
#     def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
#                  dt: float = 0.03, **kwargs):
#         super().__init__(agent, **kwargs)
#         self.config = config
#         self.steering_boundary = steering_boundary
#         self._error_buffer = deque(maxlen=10)
#         self._dt = dt
#
#     def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
#         # calculate a vector that represent where you are going
#         v_begin = self.agent.vehicle.transform.location
#         v_end = v_begin + Location(
#             x=math.cos(math.radians(self.agent.vehicle.transform.rotation.pitch)),
#             y=v_begin.y,
#             z=math.sin(math.radians(self.agent.vehicle.transform.rotation.pitch)),
#         )
#         v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, v_end.z - v_begin.z])
#
#         # calculate error projection
#         w_vec = np.array(
#             [
#                 next_waypoint.location.x - v_begin.x,
#                 next_waypoint.location.y - v_begin.y,
#                 next_waypoint.location.z - v_begin.z,
#             ]
#         )
#         _dot = math.acos(
#             np.clip(
#                 np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
#                 -1.0,
#                 1.0,
#             )
#         )
#         _cross = np.cross(v_vec, w_vec)
#         if _cross[1] > 0:
#             _dot *= -1.0
#         self._error_buffer.append(_dot)
#         if len(self._error_buffer) >= 2:
#             _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
#             _ie = sum(self._error_buffer) * self._dt
#         else:
#             _de = 0.0
#             _ie = 0.0
#
#         k_p, k_d, k_i = Stanley_controller.find_k_values(config=self.config, vehicle=self.agent.vehicle)
#
#         lat_control = float(
#             np.clip((k_p * _dot) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
#         )
#         return lat_control
#
