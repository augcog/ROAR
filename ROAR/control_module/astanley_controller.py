
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
        self.lat_stanley_controller = LatStanley_controller(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
        steering = self.lat_stanley_controller.run_in_series(next_waypoint=next_waypoint)
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

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:

        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(self.agent.vehicle.transform.rotation.pitch)),
            y=v_begin.y,
            z=math.sin(math.radians(self.agent.vehicle.transform.rotation.pitch)),
        )
        v_vec = np.array([v_end.x - v_begin.x,v_end.y - v_begin.y, v_end.z - v_begin.z])

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
        self._error_buffer.append(_dot)
       #  if len(self._error_buffer) >= 2:
       #      _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
       #      _ie = sum(self._error_buffer) * self._dt
       #  else:
       #      _de = 0.0
       #      _ie = 0.0
       #

       # # k_p, k_d, k_i = PIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)
       #
       #  lat_control = float(
       #      np.clip((.8 * _dot) + (0.3 * _de) + (0* _ie), self.steering_boundary[0], self.steering_boundary[1])
       #  )
        #return lat_control

   #********************************************************S
        veh_pitch = self.agent.vehicle.transform.rotation.pitch

        # *** referencing next waypoint coordinates ***
        path_x = next_waypoint.location.x  #*** next waypoint: self.way_points_queue[0]
        path_z = next_waypoint.location.z  #** how get

        #*** averaging path points for smooth path vector ***
        next_pathpoint1 = (self.agent.local_planner.way_points_queue[1])
        next_pathpoint2 = (self.agent.local_planner.way_points_queue[2])
        next_pathpoint3 = (self.agent.local_planner.way_points_queue[3])
        next_pathpoint4 = (self.agent.local_planner.way_points_queue[17])
        next_pathpoint5 = (self.agent.local_planner.way_points_queue[18])
        next_pathpoint6 = (self.agent.local_planner.way_points_queue[19])
        nx = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x)/6
        nz = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
        nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) /3
        nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) /3

        #***get heading if vehicle was at the correct spot on path**
        path_pitch_rad = (math.atan2((nz - path_z), (nx - path_x)))
        path_pitch = path_pitch_rad*180/np.pi

        #***difference between correct heading and actual heading - pos error gives right steering, neg gives left ***
        head_err = path_pitch - veh_pitch

        #  def run_in_series(self, next_waypoint: Transform, **kwargs) -> float: #*********** aka stanley_control(state, cx, cy, cyaw, last_target_idx)
        #      '''
        #      TODO:  tune
        #
        #      *** inputs needed: vehicle yaw, x, y; nearest path yaw, x, y
        #
        #      *** implement target calculations: heading error (vehicle yaw - path yaw at nearest path point);
        #                                         cross track error (front axle x,y - nearest path x,y)
        #
        #      self.way_points_queue[0]
        #
        #      *** output lat_control:  steering angle delta = heading error + inv tan (gain * cross track error/veh speed)
        #      '''
        #
        #
        #
        vel = self.agent.vehicle.velocity
        veh_spd = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) #*** m/s

        k = 0.5 #control gain

        # veh_loc = self.agent.vehicle.transform.location
        #***** is this in radians or angle?  guessing angle

        #pos_err, head_err = self.stan_calcs(next_waypoint)

        #lat_control = head_err + k * pos_err #if angle > 30 then 1, otherwise angle/180 ************ what does 1 equate to?  30 degrees?

        lat_control = float(
                np.clip((head_err + np.arctan(k * _dot/(veh_spd+.3)))/90, self.steering_boundary[0], self.steering_boundary[1])   #**** guessing steering of '1' equates to 30 degrees
            )

        print('_dot Stanley = ', _dot)
        print('veh_spd', veh_spd)
        print('path pitch = ', path_pitch)
        print('veh pitch = ', veh_pitch)
        print('** head_err **', head_err)
        print('lat_control = ', lat_control)
        print('-----------------------------------------')

        return lat_control
  ## ************************************************
   #
   #  def stan_calcs(self, next_waypoint: Transform, **kwargs):
   #
   #      '''
   #      calculate target
   #
   #      front axle position (corrected from veh location + lv cos and sin heading
   #      nearest point
   #
   #      front axle error (distance from front axle to desired path position)
   #
   #        front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
   #                    -np.sin(state.yaw + np.pi / 2)]
   #        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
   #      '''
   #
   #      # *** vehicle data ***
   #
   #      wb = 2.96  # assumed vehicle wheelbase (tesla)
   #
   #      veh_x = self.agent.vehicle.transform.location.x
   #      veh_y = self.agent.vehicle.transform.location.y
   #      veh_z = self.agent.vehicle.transform.location.z
   #
   #      veh_yaw = self.agent.vehicle.transform.rotation.yaw
   #      veh_roll = self.agent.vehicle.transform.rotation.roll
   #      veh_pitch = self.agent.vehicle.transform.rotation.pitch
   #
   #
   #      # *** getting front axle coordinates ***
   #      frontx = veh_x + wb*np.cos(veh_pitch*180/np.pi)/2
   #      frontz = veh_z + wb*np.sin(veh_pitch*180/np.pi)/2
   #
   #      # *** referencing next waypoint coordinates ***
   #      path_x = next_waypoint.location.x  #*** next waypoint: self.way_points_queue[0]
   #      path_z = next_waypoint.location.z  #** how get
   #
   #      #*** averaging path points for smooth path vector ***
   #      next_pathpoint1 = (self.agent.local_planner.way_points_queue[1])
   #      next_pathpoint2 = (self.agent.local_planner.way_points_queue[2])
   #      next_pathpoint3 = (self.agent.local_planner.way_points_queue[3])
   #      next_pathpoint4 = (self.agent.local_planner.way_points_queue[17])
   #      next_pathpoint5 = (self.agent.local_planner.way_points_queue[18])
   #      next_pathpoint6 = (self.agent.local_planner.way_points_queue[19])
   #      nx = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x)/6
   #      nz = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
   #      nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) /3
   #      nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) /3
   #
   #
   #      # *** calculate crosstrack error ***
   #      # *** calculate front axle position error from path with positive error = turn to right, negative = turn to left
   #
   #      dx = [frontx - nx1]
   #      dz = [frontz - nz1]
   #      # dx = [frontx - path_x]
   #      # dz = [frontz - path_z]
   #      # dx = [veh_x - path_x]
   #      # dz = [veh_z - path_z]
   #      dpathhead_rad = (math.atan2((nz1-frontz), (nx1-frontx))) # *** need some lead to get sign correct (with sin)?
   #      #dpathhead_rad = (math.atan2((path_z - frontz), (path_x - frontx)))
   #      #dpathhead_rad = (math.atan2((path_z - veh_z), (path_x - veh_x)))
   #      dpathhead_ang = dpathhead_rad * 180 / np.pi
   #      pitch_to_path = dpathhead_ang - veh_pitch
   #      dpath = np.sin(pitch_to_path*np.pi/180)*np.hypot(dx, dz) # *** pitch goes from + to - as crosses x axis
   #
   #      # dpath = np.hypot(dx, dz)-8  #  really should take this value * sign of pitch_to_path
   #
   #
   #      # front_axle_vec = [-np.cos(veh_yaw + np.pi / 2), -np.sin(veh_yaw + np.pi / 2)]   # RMS error?
   #      # e_front_axle_pos = np.dot([nx, ny], front_axle_vec)
   #
   #      #***get heading if vehicle was at the correct spot on path**
   #      path_pitch_rad = (math.atan2((nz - path_z), (nx - path_x)))
   #      path_pitch = path_pitch_rad*180/np.pi
   #
   #      #***difference between correct heading and actual heading - pos error gives right steering, neg gives left ***
   #      head_err = path_pitch - veh_pitch
   #
   #      #*************************************  borrowed from pid for _dot
   #      # calculate a vector that represent where you are going
   #      v_begin = self.agent.vehicle.transform.location
   #      v_end = v_begin + Location(
   #          x=math.cos(math.radians(self.agent.vehicle.transform.rotation.pitch)),
   #          y=v_begin.y,
   #          z=math.sin(math.radians(self.agent.vehicle.transform.rotation.pitch)),
   #      )
   #      v_vec = np.array([v_end.x - v_begin.x,v_end.y - v_begin.y, v_end.z - v_begin.z])
   #
   #      # calculate error projection
   #      w_vec = np.array(
   #          [
   #              next_waypoint.location.x - v_begin.x,
   #              next_waypoint.location.y - v_begin.y,
   #              next_waypoint.location.z - v_begin.z,
   #          ]
   #      )
   #      _dot = math.acos(
   #          np.clip(
   #              np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)),
   #              -1.0,
   #              1.0,
   #          )
   #      )
   #      _cross = np.cross(v_vec, w_vec)
   #      if _cross[1] > 0:
   #          _dot *= -1.0
   #      #***************************************
   #
   #      print('--------------------------------------')
   #      # print('veh yaw = ', veh_yaw)
   #      # print('veh roll = ', veh_roll)
   #      print('veh pitch = ', veh_pitch)
   #      print('**pitch to path** = ', pitch_to_path)
   #      #
   #      print('veh x = ', veh_x)
   #      print('veh z = ', veh_z)
   #      # print('front x = ', frontx)
   #      # print('front z = ', frontz)
   #      print('path x = ', path_x)
   #      print('path z = ', path_z)
   #      print('next path x = ', nx)
   #      print('next path z = ', nz)
   #      print('**distance to path = ', dpath)
   #      print('path pitch = ', path_pitch)
   #      print('path_pitch_rad = ', path_pitch_rad)
   #      # print('path queue 0 = ', self.agent.local_planner.way_points_queue[0])
   #      # print('path queue 4 = ', self.agent.local_planner.way_points_queue[9])
   #      # print('path queue 20 = ', self.agent.local_planner.way_points_queue[17])
   #      print('** heading error **', head_err)
   #      print('_dot err', _dot)
   #
   #      return _dot, head_err

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
