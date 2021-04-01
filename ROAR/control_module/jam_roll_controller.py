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
       #self.max_speed = self.agent.agent_settings.max_speed
        self.max_speed = 170 #************************* MAX SPEED *********************************

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
        # print('roll: ', veh_roll)
        # print('pitch: ', veh_pitch)

        return VehicleControl(throttle=throttle, steering=steering)

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        #k_p, k_d, k_i = .03, 0.9, 0  #original values
        k_p, k_d, k_i = .1, 4, 0

        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.clip([k_p, k_d, k_i], a_min=0, a_max=1)

# *** Roll ContRoller v2 ***
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
# **************************

# *** original Roll ContRoller + v2 ***
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

        print('max speed: ',self.max_speed)

        k_p, k_d, k_i = PIDController.find_k_values(vehicle=self.agent.vehicle, config=self.config)
        error = target_speed - current_speed

        self._error_buffer.append(error)


        #****************** implement look ahead *******************
        la_err = self.la_calcs(next_waypoint)
        kla = .02

        if len(self._error_buffer) >= 2:
            # print(self._error_buffer[-1], self._error_buffer[-2])
            _de = (self._error_buffer[-2] - self._error_buffer[-1]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        # output = float(np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.throttle_boundary[0],
        #                        self.throttle_boundary[1]))
        print(self.agent.vehicle.transform.rotation.roll)
        vehroll = self.agent.vehicle.transform.rotation.roll
        if current_speed >= (target_speed + 2):
            out = 1 - .1 * (current_speed - target_speed)
        else:
            if abs(self.agent.vehicle.transform.rotation.roll) <= .35:
                out = 6 * np.exp(-0.05 * np.abs(vehroll))-(la_err/180)*current_speed*kla
            else:
                out = 2 * np.exp(-0.05 * np.abs(vehroll))-(la_err/180)*current_speed*kla  # *****ALGORITHM*****

        output = np.clip(out, a_min=0, a_max=1)
        print('*************')
        print('throttle = ', output)
        print('*************')

        # if abs(self.agent.vehicle.transform.rotation.roll) <= .35:
        #     output = 1
        #     if abs(self.agent.vehicle.transform.rotation.roll) > .35:
        #           # output = 1.2*np.exp(-0.07 * np.abs(vehroll))
        #           # output = 4 * np.exp(-0.06 * np.abs(vehroll))
        #
        #         output = 0
        #         if abs(self.agent.vehicle.transform.rotation.roll) > .6:
        #             output = .8
        #             if abs(self.agent.vehicle.transform.rotation.roll) > 1.2:
        #                 output = .7
        #                 if abs(self.agent.vehicle.transform.rotation.roll) > 1.5:
        #                     output = 1/(3.1**(self.agent.vehicle.transform.rotation.roll))
        #                     if abs(self.agent.vehicle.transform.rotation.roll) > 7:
        #                         output = 0
        #                 if abs(self.agent.vehicle.transform.rotation.roll) > 1:
        #                     output = .7
        #                     if abs(self.agent.vehicle.transform.rotation.roll) > 3:
        #                         output = .4
        #                         if abs(self.agent.vehicle.transform.rotation.roll) > 4:
        #                             output = .2
        #                             if abs(self.agent.vehicle.transform.rotation.roll) > 6:
        #                                 output = 0

        # self.logger.debug(f"curr_speed: {round(current_speed, 2)} | kp: {round(k_p, 2)} | kd: {k_d} | ki = {k_i} | "
        #       f"err = {round(error, 2)} | de = {round(_de, 2)} | ie = {round(_ie, 2)}")
        #       f"self._error_buffer[-1] {self._error_buffer[-1]} | self._error_buffer[-2] = {self._error_buffer[-2]}")
        return output

    def la_calcs(self, next_waypoint: Transform, **kwargs):

        current_speed = int(Vehicle.get_speed(self.agent.vehicle))
        cs = np.clip(current_speed, 70, 200)
        # *** next points on path
        # *** averaging path points for smooth path vector ***


        # next_pathpoint1 = (self.agent.local_planner.way_points_queue[2*cs+1])
        # next_pathpoint2 = (self.agent.local_planner.way_points_queue[2*cs+2])
        # next_pathpoint3 = (self.agent.local_planner.way_points_queue[2*cs+3])
        # next_pathpoint4 = (self.agent.local_planner.way_points_queue[2*cs+91])
        # next_pathpoint5 = (self.agent.local_planner.way_points_queue[2*cs+92])
        # next_pathpoint6 = (self.agent.local_planner.way_points_queue[2*cs+93])
        next_pathpoint1 = (self.agent.local_planner.way_points_queue[3*cs+1])
        next_pathpoint2 = (self.agent.local_planner.way_points_queue[3*cs+2])
        next_pathpoint3 = (self.agent.local_planner.way_points_queue[3*cs+3])
        next_pathpoint4 = (self.agent.local_planner.way_points_queue[3*cs+71])
        next_pathpoint5 = (self.agent.local_planner.way_points_queue[3*cs+72])
        next_pathpoint6 = (self.agent.local_planner.way_points_queue[3*cs+73])
        # next_pathpoint4 = (self.agent.local_planner.way_points_queue[cs+43])
        # next_pathpoint5 = (self.agent.local_planner.way_points_queue[cs+42])
        # next_pathpoint6 = (self.agent.local_planner.way_points_queue[cs+41])
        # next_pathpoint1 = (self.agent.local_planner.way_points_queue[31])
        # next_pathpoint2 = (self.agent.local_planner.way_points_queue[32])
        # next_pathpoint3 = (self.agent.local_planner.way_points_queue[33])
        # next_pathpoint4 = (self.agent.local_planner.way_points_queue[52])
        # next_pathpoint5 = (self.agent.local_planner.way_points_queue[53])
        # next_pathpoint6 = (self.agent.local_planner.way_points_queue[54])
        nx0 = next_pathpoint1.location.x
        nz0 = next_pathpoint1.location.z
        nx = (
                         next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 6
        nz = (
                         next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
        nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) / 3
        nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) / 3
        nx2 = (next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 3
        nz2 = (next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 3

        npath0 = np.transpose(np.array([nx0, nz0, 1]))
        npath = np.transpose(np.array([nx, nz, 1]))
        npath1 = np.transpose(np.array([nx1, nz1, 1]))
        npath2 = np.transpose(np.array([nx2, nz2, 1]))

        path_yaw_rad = (math.atan2((nx2 - nx1), -(nz2 - nz1)))

        path_yaw = path_yaw_rad * 180 / np.pi
        print(' !!! path yaw !!! ', path_yaw)

        veh_yaw = self.agent.vehicle.transform.rotation.yaw
        print(' !!! veh yaw  !!! ', veh_yaw)
        ahead_err = abs(abs(path_yaw)-abs(veh_yaw))
        if ahead_err < 70:
            la_err = 0
        else:
            la_err =(.05 * ahead_err)**3
        # if ahead_err < 75:
        #     la_err = 0
        # elif ahead_err > 86:
        #     la_err = 3 * ahead_err
        # else:
        #     la_err = 2 * ahead_err

        # if la_err > 180:
        #     ahead_err = la_err - 360
        # elif la_err < -180:
        #     ahead_err = la_err + 360
        # else:
        #     ahead_err = la_err

        print('--------------------------------------')

        # print(f"{veh_x},{veh_y},{veh_z},{veh_roll},{veh_pitch},{veh_yaw}")
        # datarow = f"{veh_x},{veh_y},{veh_z},{veh_roll},{veh_pitch},{veh_yaw}"
        # self.waypointrecord.append(datarow.split(","))

        print('** la err **', la_err)
        print('--------------------------------------')
        #
        # print('** look ahead error **', ahead_err)



        return la_err


        #***********************************************************

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

        # print(v_begin)
        # print('next wp x: ', next_waypoint.location.x)
        # print('next wp z: ', next_waypoint.location.z)
        # print('next wp y: ', next_waypoint.location.y)

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

        hed_err = self.hd_calc(next_waypoint)
        kle = 0.1
        k_p, k_d, k_i = PIDController.find_k_values(config=self.config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie) + (kle * hed_err), self.steering_boundary[0], self.steering_boundary[1])
        )
        # print(f"v_vec_normed: {v_vec_normed} | w_vec_normed = {w_vec_normed}")
        # print("v_vec_normed @ w_vec_normed.T:", v_vec_normed @ w_vec_normed.T)
        # print(f"Curr: {self.agent.vehicle.transform.location}, waypoint: {next_waypoint}")
        # print(f"lat_control: {round(lat_control, 3)} | error: {error} ")
        # print()
        return lat_control

    def hd_calc(self, next_waypoint: Transform, **kwargs):

        # *** get vehicle location info ***
        veh_x = self.agent.vehicle.transform.location.x
        veh_y = self.agent.vehicle.transform.location.y
        veh_z = self.agent.vehicle.transform.location.z

        # *** create world to veh transformation ***
        veh_yaw = self.agent.vehicle.transform.rotation.yaw
        theta_deg = veh_yaw
        theta_rad = np.radians(theta_deg)
        gwv = np.array([[np.cos(theta_rad), -np.sin(theta_rad), veh_z],
                        [np.sin(theta_rad), np.cos(theta_rad), veh_x],
                        [0, 0, 1]])

        gvw = np.linalg.inv(gwv)

        # *** next points on path
        # *** averaging path points for smooth path vector ***
        next_pathpoint1 = (self.agent.local_planner.way_points_queue[1])
        next_pathpoint2 = (self.agent.local_planner.way_points_queue[2])
        next_pathpoint3 = (self.agent.local_planner.way_points_queue[3])
        next_pathpoint4 = (self.agent.local_planner.way_points_queue[17])
        next_pathpoint5 = (self.agent.local_planner.way_points_queue[18])
        next_pathpoint6 = (self.agent.local_planner.way_points_queue[19])
        nx0 = next_pathpoint1.location.x
        nz0 = next_pathpoint1.location.z
        nx = (
                         next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x + next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 6
        nz = (
                         next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z + next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 6
        nx1 = (next_pathpoint1.location.x + next_pathpoint2.location.x + next_pathpoint3.location.x) / 3
        nz1 = (next_pathpoint1.location.z + next_pathpoint2.location.z + next_pathpoint3.location.z) / 3
        nx2 = (next_pathpoint4.location.x + next_pathpoint5.location.x + next_pathpoint6.location.x) / 3
        nz2 = (next_pathpoint4.location.z + next_pathpoint5.location.z + next_pathpoint6.location.z) / 3

        # *** convert path points to veh frame ***
        npath0 = np.transpose(np.array([nz0, nx0, 1]))
        npath = np.transpose(np.array([nz, nx, 1]))
        npath1 = np.transpose(np.array([nz1, nx1, 1]))
        npath2 = np.transpose(np.array([nz2, nx2, 1]))

        vf_npath0 = np.matmul(gvw, npath0)
        vf_npath = np.matmul(gvw, npath)
        vf_npath1 = np.matmul(gvw, npath1)
        vf_npath2 = np.matmul(gvw, npath2)
        print ('veh frame path z value: ',vf_npath[0])
        print ('veh frame path x value: ',vf_npath[1])

        # *** convert path points to yaw error in veh frame ***

        path_yaw = math.atan2(-(vf_npath2[1]-vf_npath1[1]),-(vf_npath2[0]-vf_npath1[0]))
        head_err = -np.rad2deg(path_yaw)/180 # - because we want positive yaw to to turn left which is negative vice versa
        # hd_err = path_yaw * 180 / np.pi
        # head_err = 0
        # if hd_err > 180:
        #     head_err = hd_err - 360
        # elif hd_err < -180:
        #     head_err = hd_err + 360
        # else:
        #     head_err = hd_err

        print('** heading error **', head_err)

        return head_err
