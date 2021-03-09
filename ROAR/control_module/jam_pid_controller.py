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
       # print(        self.agent.vehicle.transform.rotation.roll)
        print('steering', steering)

        veh_x = self.agent.vehicle.transform.location.x
        veh_y = self.agent.vehicle.transform.location.y
        veh_z = self.agent.vehicle.transform.location.z

        veh_yaw = self.agent.vehicle.transform.rotation.yaw
        veh_roll = self.agent.vehicle.transform.rotation.roll
        veh_pitch = self.agent.vehicle.transform.rotation.pitch

        print('pos x: ', veh_x)
        print('pos y: ', veh_y)
        print('pos z: ', veh_z)

        print('yaw: ', veh_yaw)
        print('roll: ', veh_roll)
        print('pitch: ', veh_pitch)

        return VehicleControl(throttle=throttle, steering=steering)

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = .03, 0.9, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.clip([k_p, k_d, k_i], a_min=0, a_max=1)

# ***** new version Roll *****
# class LongPIDController(Controller):
#     def __init__(self, agent, config: dict, throttle_boundary: Tuple[float, float], max_speed: float,
#                  dt: float = 0.03, **kwargs):
#         super().__init__(agent, **kwargs)
#         self.config = config
#         self.max_speed = max_speed
#         self.throttle_boundary = throttle_boundary
#         self._error_buffer = deque(maxlen=10)
#         self.gpd = GroundPlaneDetector(agent, vis=True)
#         self.cpd = ColorPlaneDetector(agent)
#         self._dt = dt
#         self.time_step = 0
#
#     def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
#         target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed))
#         current_speed = Vehicle.get_speed(self.agent.vehicle)
#
#         k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config=self.config)
#         error = target_speed - current_speed
#
#         self._error_buffer.append(error)
#
#         if len(self._error_buffer) >= 2:
#             # print(self._error_buffer[-1], self._error_buffer[-2])
#             _de = abs((self._error_buffer[-2] - self._error_buffer[-1])) / self._dt
#             _ie = sum(self._error_buffer) * self._dt
#         else:
#             _de = 0.0
#             _ie = 0.0
#         output = float(np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.throttle_boundary[0],
#                                self.throttle_boundary[1]))
#
#         if self.time_step % 5 == 0:
#             # self.gpd.run_in_series()
#             self.gpd.run_in_series()
#             road_normal = np.array(self.gpd.road_normal)
#             road_dot = np.dot(road_normal, np.array([0, 1, 0]))
#             true_roll = self.agent.vehicle.transform.rotation.roll
#             pred_roll = np.arccos(road_dot)
#
#             # print(road_normal)
#             # print(np.arccos(road_dot))
#             # print(self.agent.vehicle.transform.rotation.roll)
#             # print(np.abs(true_roll - pred_roll))
#
#             self.time_step = self.time_step % 5
#         self.time_step += 1
#
#         use_plane_roll = True
#         pred_roll = self.agent.vehicle.transform.rotation.roll
#         if use_plane_roll:
#             pred_roll = self.gpd.pred_roll
#
#         output = np.exp(-0.048 * np.abs(pred_roll))
#
#         # self.logger.debug(f"curr_speed: {round(current_speed, 2)} | kp: {round(k_p, 2)} | kd: {k_d} | ki = {k_i} | "
#         #       f"err = {round(error, 2)} | de = {round(_de, 2)} | ie = {round(_ie, 2)}")
#               #f"self._error_buffer[-1] {self._error_buffer[-1]} | self._error_buffer[-2] = {self._error_buffer[-2]}")
#         return output
#

# ***** original version *****
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
        print(self.agent.vehicle.transform.rotation.roll)
        vehroll=self.agent.vehicle.transform.rotation.roll
        # output = np.exp(-0.048 * np.abs(vehroll))
        if abs(self.agent.vehicle.transform.rotation.roll) <= .75:
            output = 1
            if abs(self.agent.vehicle.transform.rotation.roll) > .75:
                  output = np.exp(-0.06 * np.abs(vehroll))
        #         output = 0
        #         if abs(self.agent.vehicle.transform.rotation.roll) > .6:
        #             output = .8
        #             if abs(self.agent.vehicle.transform.rotation.roll) > 1.2:
        #                 output = .7
        #                 if abs(self.agent.vehicle.transform.rotation.roll) > 1.5:
        #                     output = 1/(3.1**(self.agent.vehicle.transform.rotation.roll))
        #                     if abs(self.agent.vehicle.transform.rotation.roll) > 7:
        #                         output = 0
                    #     if abs(self.agent.vehicle.transform.rotation.roll) > 1:
                    #         output = .7
                    #         if abs(self.agent.vehicle.transform.rotation.roll) > 3:
                    #             output = .4
                    #             if abs(self.agent.vehicle.transform.rotation.roll) > 4:
                    #                 output = .2
                    #                 if abs(self.agent.vehicle.transform.rotation.roll) > 6:
                    #                     output = 0

        # self.logger.debug(f"curr_speed: {round(current_speed, 2)} | kp: {round(k_p, 2)} | kd: {k_d} | ki = {k_i} | "
        #       f"err = {round(error, 2)} | de = {round(_de, 2)} | ie = {round(_ie, 2)}")
              #f"self._error_buffer[-1] {self._error_buffer[-1]} | self._error_buffer[-2] = {self._error_buffer[-2]}")
        return output
# ***** end original version Roll ContRoller *****

class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
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
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        print(v_begin)
        print('next wp x: ', next_waypoint.location.x)
        print('next wp z: ', next_waypoint.location.z)
        print('next wp y: ', next_waypoint.location.y)


        direction_vector = np.array([np.sin(np.radians(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.radians(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([v_end[0] - v_begin[0], 0, - (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        _dot = math.acos(np.clip(np.dot(v_vec, w_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[1] < 0:
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
        print(f"lat_control: {round(lat_control, 3)} | v_vec = {v_vec} | w_vec = {w_vec}")
        return lat_control
# class LatPIDController(Controller):
#
#     def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
#                  dt: float = 0.03, **kwargs):
#         super().__init__(agent, **kwargs)
#         self.config = config
#         self.steering_boundary = steering_boundary
#         self._error_buffer = deque(maxlen=10)
#         self._dt = dt
#         self.waypointrecord = []
#
#     def run_in_series(self, next_waypoint: Transform,
#                       **kwargs) -> float:  # *********** aka stanley_control(state, cx, cy, cyaw, last_target_idx)
#         '''
#         TODO:  tune
#
#         *** inputs needed: vehicle yaw, x, y; nearest path yaw, x, y
#
#         *** implement target calculations: heading error (vehicle yaw - path yaw at nearest path point);
#                                            cross track error (front axle x,y - nearest path x,y)
#
#         self.way_points_queue[0]
#
#         *** output lat_control:  steering angle delta = heading error + inv tan (gain * cross track error/veh speed)
#         '''
#
#         vel = self.agent.vehicle.velocity
#         veh_spd = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # *** m/s
#
#         k = 1  # control gain
#
#         # veh_loc = self.agent.vehicle.transform.location
#
#         pos_err, head_err = self.stan_calcs(next_waypoint)
#
#         # lat_control = head_err + k * pos_err #if angle > 30 then 1, otherwise angle/180 ************ what does 1 equate to?  30 degrees?
#
#         lat_control = float(
#             np.clip((.11 * head_err + (np.arctan(k * pos_err / (veh_spd + .3)))) / 30, self.steering_boundary[0],
#                     self.steering_boundary[1])  # **** guessing steering of '1' equates to 30 degrees
#         )
#         print('-----------------------------------------')
#
#         print('bStanley lat_control = ', lat_control)
#         print('-----------------------------------------')
#
#         return lat_control
#
#     def stan_calcs(self, next_waypoint: Transform, **kwargs):
#
#         '''
#         calculate target
#
#         front axle position (corrected from veh location + lv cos and sin heading
#         nearest point
#
#         front axle error (distance from front axle to desired path position)
#
#           front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
#                       -np.sin(state.yaw + np.pi / 2)]
#           error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)
#         '''
#
#         # *** vehicle data ***
#
#         wb = 2.96  # assumed vehicle wheelbase (tesla)
#
#         veh_x = self.agent.vehicle.transform.location.x
#         veh_y = self.agent.vehicle.transform.location.y
#         veh_z = self.agent.vehicle.transform.location.z
#
#         veh_yaw = self.agent.vehicle.transform.rotation.yaw
#         veh_roll = self.agent.vehicle.transform.rotation.roll
#         veh_pitch = self.agent.vehicle.transform.rotation.pitch
#
#         print('pos x: ', veh_x)
#         print('pos y: ', veh_y)
#         print('pos z: ', veh_z)
#
#         print('yaw: ', veh_yaw)
#         print('pitch: ', veh_pitch)
#         print('roll: ', veh_roll)
#
#         # ************* convert points to vehicle reference *****************
#
#         theta_deg = veh_yaw
#         theta_rad = np.radians(theta_deg)
#         # gvw3d=np.array([[np.cos (theta_rad), 0, np.sin (theta_rad)],
#         #              [0,          1,        0 ],
#         #              [-np.sin (theta_rad), 0, np.cos (theta_rad)]])
#         # gwv = np.array([[np.cos(theta_rad), -np.sin(theta_rad), veh_x],
#         #                 [np.sin(theta_rad), np.cos(theta_rad), veh_z],
#         #                 [0, 0, 1]])
#
#         gwv = np.array([[np.cos(theta_rad), np.sin(theta_rad), veh_x],
#                         [-np.sin(theta_rad), np.cos(theta_rad), veh_z],
#                         [0, 0, 1]])
#
#         gvw = np.linalg.inv(gwv)
#         # *** define points in vehicle reference frame ***
#
#         # *** vehicle front axle ***
#         fx = 0
#         fz = -.5 * wb
#
#         # *** next waypoint ***
#
#         nextwp = np.transpose(np.array([next_waypoint.location.x, next_waypoint.location.z, 1]))
#         vf_nextwp = np.matmul(gvw, nextwp)
#
#         # nextwp = np.transpose(np.array([next_waypoint.location.x, next_waypoint.location.z, 1]))
#         # vf_nextwp = np.matmul(gvw, nextwp)
#
#         # *** next points on path
#         # *** averaging path points for smooth path vector ***
#         next_pathpoint1 = (self.agent.local_planner.way_points_queue[1])
#         next_pathpoint2 = (self.agent.local_planner.way_points_queue[2])
#         next_pathpoint3 = (self.agent.local_planner.way_points_queue[3])
#         next_pathpoint4 = (self.agent.local_planner.way_points_queue[17])
#         next_pathpoint5 = (self.agent.local_planner.way_points_queue[18])
#         next_pathpoint6 = (self.agent.local_planner.way_points_queue[19])
#         nx0 = next_pathpoint1.location.x
#         nz0 = next_pathpoint1.location.z
#         nx = (
#                          next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 6
#         nz = (
#                          next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
#         nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) / 3
#         nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) / 3
#         nx2 = (next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 3
#         nz2 = (next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 3
#
#         npath0 = np.transpose(np.array([nx0, nz0, 1]))
#         npath = np.transpose(np.array([nx, nz, 1]))
#         npath1 = np.transpose(np.array([nx1, nz1, 1]))
#         npath2 = np.transpose(np.array([nx2, nz2, 1]))
#
#         vf_npath0 = np.matmul(gvw, npath0)
#         vf_npath = np.matmul(gvw, npath)
#         vf_npath1 = np.matmul(gvw, npath1)
#         vf_npath2 = np.matmul(gvw, npath2)
#
#         # ***
#         # print ('theta_deg = ',theta_deg)
#         # print ('vf_npath = ',vf_npath)
#         # print('vf_npath1 = ', vf_npath1)
#         # print('vf_npath2 = ', vf_npath2)
#         # print ('Gvw = ', gvw)
#         # # print ('Gvw inv = ', gwv)
#         print('vehicle frame next waypoint = ', vf_nextwp)
#         print('next waypoint = ', nextwp)
#         # print('next waypoint object',next_waypoint)
#         # print ('next waypoint = ', next_waypoint.location)
#         '''
#         # *** get in vehicle reference ***
#
#         path coordinates
#         next_wp
#
#         vehicle coordinates
#
#         '''
#
#         # # *** getting front axle coordinates ***
#         # frontx = veh_x + wb*np.cos(veh_pitch*180/np.pi)/2
#         # frontz = veh_z + wb*np.sin(veh_pitch*180/np.pi)/2
#
#         # # *** referencing next waypoint coordinates ***
#         # path_x = next_waypoint.location.x  #*** next waypoint: self.way_points_queue[0]
#         # path_z = next_waypoint.location.z  #** how get
#
#         # *** calculate crosstrack error ***
#         # *** calculate front axle position error from path with positive error = turn to right, negative = turn to left
#         vf_cterad = 1 * math.atan2(vf_npath0[0], vf_npath0[1])
#         vf_ctedeg = vf_cterad * 180 / math.pi
#         vf_cte = vf_ctedeg
#
#         # vf_cte = vf_npath0[0]
#
#         # dx = nx1-frontx
#         # dz = nz1-frontz
#
#         # dx = veh_x - path_x
#         # dz = veh_z - path_z
#         #         dpathhead_rad = (math.atan2(dz, dx)) # *** need some lead to get sign correct (with sin)?
#         #         #dpathhead_rad = (math.atan2((path_z - frontz), (path_x - frontx)))
#         #         #dpathhead_rad = (math.atan2((path_z - veh_z), (path_x - veh_x)))
#         #         dpathhead_ang = dpathhead_rad * 180 / np.pi
#         #         pitch_to_path = dpathhead_ang - veh_pitch
#         #         #dpath = np.cos(pitch_to_path*np.pi/180)*np.hypot(dx, dz) # *** pitch goes from + to - as crosses x axis
#         #
#         #         #dpath = np.hypot(dx, dz)-8  #  really should take this value * sign of pitch_to_path
#         # #*************************
#         #
#         #         #dpath = (((nx2-path_x)*(path_z-veh_z)-(path_x-veh_x)*(nz2-path_z)))/(((nx2-path_x)**2+(nz2-path_z)**2)**.5)
#         #         dpath = (((nx2-nx1)*(nz1-veh_z)-(nx1-veh_x)*(nz2-nz1)))/(((nx2-nx1)**2+(nz2-nz1)**2)**.5)
#
#         # *************************
#
#         # front_axle_vec = [-np.cos(veh_yaw + np.pi / 2), -np.sin(veh_yaw + np.pi / 2)]   # RMS error?
#         # e_front_axle_pos = np.dot([nx, ny], front_axle_vec)
#
#         # ***get heading if vehicle was at the correct spot on path**
#         # vf_path_pitch = np.degrees(math.atan2((vf_npath2[1] - vf_nextwp[1]), (vf_npath2[0] - vf_nextwp[0])))
#
#         path_yaw_rad = (math.atan2((nx2 - next_waypoint.location.x), (nz2 - next_waypoint.location.z)))
#         path_yaw = path_yaw_rad * 180 / np.pi
#
#         # ***difference between correct heading and actual heading - pos error gives right steering, neg gives left ***
#         hd_err = path_yaw - veh_yaw
#         head_err = 0
#         if hd_err > 180:
#             head_err = hd_err - 360
#         elif hd_err < -180:
#             head_err = hd_err + 360
#         else:
#             head_err = hd_err
#
#         # head_err = vf_path_pitch - theta_deg
#
#         print('--------------------------------------')
#         print('veh yaw = ', veh_yaw)
#         # # print('veh roll = ', veh_roll)
#         # print('veh pitch = ', veh_pitch)
#         # print('**pitch to path** = ', pitch_to_path)
#         # print('**dpathhead_ang**',dpathhead_ang)
#         # print('veh x = ', veh_x)
#         # print('veh z = ', veh_z)
#
#         print(f"{veh_x},{veh_y},{veh_z},{veh_roll},{veh_pitch},{veh_yaw}")
#         datarow = f"{veh_x},{veh_y},{veh_z},{veh_roll},{veh_pitch},{veh_yaw}"
#         self.waypointrecord.append(datarow.split(","))
#
#         # # print('front x = ', frontx)
#         # # print('front z = ', frontz)
#         # print('path x = ', path_x)
#         # print('path z = ', path_z)
#         # print('next path x1 = ', nx1)
#         # print('next path z1 = ', nz1)
#         # print('next path x = ', nx)
#         # print('next path z = ', nz)
#         # print('next path x2 = ', nx2)
#         # print('next path z2 = ', nz2)
#         #
#         # print('**distance to path = ', dpath)
#         print('path pitch = ', path_yaw)
#         # print('path_pitch_rad = ', path_pitch_rad)
#         # # print('path queue 0 = ', self.agent.local_planner.way_points_queue[0])
#         # # print('path queue 4 = ', self.agent.local_planner.way_points_queue[9])
#         # # print('path queue 20 = ', self.agent.local_planner.way_points_queue[17])
#         print('** hd err **', hd_err)
#
#         print('** heading error **', head_err)
#         print('vf cross track error', vf_cte)
#         # # print('_dot err', _dot)
#         # vf_cte=0  # check if goes straight
#         # head_err=0
#
#         return vf_cte, head_err
#
#         '''
#         *** end my code ***
#         '''
#
