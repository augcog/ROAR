# Copyright (c) # Copyright (c) 2018-2020 CVC.
# Modified version from Carla PID controller
# https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/controller.py
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """
from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import Transform, Location
from collections import deque
import numpy as np
import math
import logging
from ROAR.agent_module.agent import Agent


class PIDParam(BaseModel):
    K_P: float = Field(default=1)
    K_D: float = Field(default=1)
    K_I: float = Field(default=1)
    dt: float = Field(default=1)

    @staticmethod
    def default_lateral_param():
        return PIDParam(K_P=1.95, K_D=0.2, K_I=0.07, dt=1.0 / 20.0)

    @staticmethod
    def default_longitudinal_param():
        return PIDParam(K_P=1, K_D=0, K_I=0.05, dt=1.0 / 20.0)


class VehiclePIDController(Controller):
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(
            self,
            agent: Agent,
            args_lateral: PIDParam = PIDParam,
            args_longitudinal: PIDParam = PIDParam,
            target_speed=float("inf"),
            max_throttle=1,
            max_steering=1,
    ):
        """

        Args:
            agent   : actor to apply to local planner logic onto
            args_lateral:  dictionary of arguments to set the lateral PID control
            args_longitudinal: dictionary of arguments to set the longitudinal
            target_speed: target speedd in km/h
            max_throttle: maximum throttle from, will be capped at 1
            max_steering: absolute maximum steering ranging from -1 - 1
        """

        super().__init__(agent=agent)
        self.logger = logging.getLogger(__name__)
        self.max_throttle = max_throttle
        self.max_steer = max_steering

        self.target_speed = target_speed

        self.past_steering = self.agent.vehicle.control.steering
        self._lon_controller = PIDLongitudinalController(
            self.agent,
            K_P=args_longitudinal.K_P,
            K_D=args_longitudinal.K_D,
            K_I=args_longitudinal.K_I,
            dt=args_longitudinal.dt,
        )
        self._lat_controller = PIDLateralController(
            self.agent,
            K_P=args_lateral.K_P,
            K_D=args_lateral.K_D,
            K_I=args_lateral.K_I,
            dt=args_lateral.dt,
        )
        self.logger.debug("PID Controller initiated")

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint at a given target_speed.

        Args:
            vehicle: New vehicle state
            next_waypoint:  target location encoded as a waypoint
            **kwargs:

        Returns:
            Next Vehicle Control
        """

        self.update_lon_control_k_values()
        self.update_lat_control_k_values()

        acceleration = self._lon_controller.run_step(self.target_speed)
        current_steering = self._lat_controller.run_step(next_waypoint)
        control = VehicleControl()

        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throttle)
            # control.brake = 0.0
        else:
            control.throttle = 0
            # control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steering = steering
        self.past_steering = steering
        abs_throttle = 0.2
        return self.throttle_regularization(control=control, min_throttle=-abs_throttle, max_throttle=abs_throttle)

    def sync_data(self) -> None:
        self._lon_controller.vehicle = self.agent.vehicle
        self._lat_controller.vehicle = self.agent.vehicle

    def throttle_regularization(self, control: VehicleControl, min_throttle, max_throttle):
        control.throttle = np.interp(x=control.throttle, xp=[-1, 1], fp=[min_throttle, max_throttle])
        return control

    def update_lon_control_k_values(self):
        curr_speed = Vehicle.get_speed(self.agent.vehicle)
        keys = list(OPTIMIZED_LONGITUDINAL_PID_VALUES.keys())
        for speed_boundary in keys:
            if curr_speed <= speed_boundary:
                pid = OPTIMIZED_LONGITUDINAL_PID_VALUES[speed_boundary]
                self._lon_controller._k_p, self._lon_controller._k_d, self._lon_controller._k_i, \
                self._lon_controller._dt = pid.K_P, pid.K_D, pid.K_I, pid.dt
                break
        # print(self._lon_controller._k_p, self._lon_controller._k_d, self._lon_controller._k_i)

    def update_lat_control_k_values(self):
        curr_speed = Vehicle.get_speed(self.agent.vehicle)
        keys = list(OPTIMIZED_LATERAL_PID_VALUES.keys())
        # print(curr_speed)
        for speed_boundary in keys:
            if curr_speed <= speed_boundary:
                pid = OPTIMIZED_LATERAL_PID_VALUES[speed_boundary]
                self._lat_controller.k_p, self._lat_controller.k_d, self._lat_controller.k_i, \
                self._lat_controller._dt = pid.K_P, pid.K_D, pid.K_I, pid.dt
                break
        # print(self._lat_controller.k_p, self._lat_controller.k_d, self._lat_controller.k_i)

# speed - LONGITUDINALPIDParam
OPTIMIZED_LONGITUDINAL_PID_VALUES = {
    0.1: PIDParam(K_P=1, K_D=0.2, K_I=0),
    0.2: PIDParam(K_P=0.9, K_D=0.2, K_I=0),
    0.5: PIDParam(K_P=0.8, K_D=0.2, K_I=0),
    2.0: PIDParam(K_P=0.7, K_D=0.2, K_I=0),
    3.0: PIDParam(K_P=0.4, K_D=0.5, K_I=0),
    10.0: PIDParam(K_P=0, K_D=0, K_I=0),  # will never reach this speed
}

# speed - LATERALPIDParam
OPTIMIZED_LATERAL_PID_VALUES = {
    0.1: PIDParam(K_P=0.5, K_D=0.5, K_I=0),
    0.2: PIDParam(K_P=0.5, K_D=0.5, K_I=0),
    10: PIDParam(K_P=0.5, K_D=0.5, K_I=0),  # will never reach this speed
}


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, agent: Agent, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.
            :param agent: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self.agent = agent
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed):
        """
        Execute one step of longitudinal control to reach a given target speed.
            :param target_speed: target speed in Km/h
            :return: throttle control
        """
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed) -> float:
        """
        Estimate the throttle/brake of the vehicle based on the PID equations
            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        output = float(
            np.clip(
                (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0
            )
        )
        # print(output, self._k_p, self._k_d, self._k_i,self._k_p * error, self._k_d * _de, self._k_i * _ie, error)
        return output


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, agent: Agent, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.
            :param agent: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self.agent: Agent = agent
        self.k_p = K_P
        self.k_d = K_D
        self.k_i = K_I
        self.dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_waypoint: Transform) -> float:
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.
            :param target_waypoint:
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(
            target_waypoint=target_waypoint, vehicle_transform=self.agent.vehicle.transform
        )

    def _pid_control(self, target_waypoint, vehicle_transform) -> float:
        """
        Estimate the steering angle of the vehicle based on the PID equations
            :param target_waypoint: target waypoint
            :param vehicle_transform: current transform of the vehicle
            :return: steering control in the range [-1, 1]
        """
        # calculate a vector that represent where you are going
        v_begin = vehicle_transform.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
            y=math.sin(math.radians(vehicle_transform.rotation.yaw)),
            z=0,
        )
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

        # calculate error projection
        w_vec = np.array(
            [
                target_waypoint.location.x - v_begin.x,
                target_waypoint.location.z - v_begin.z,
                0.0,
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

        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return float(
            np.clip((self.k_p * _dot) + (self.k_d * _de) + (self.k_i * _ie), -1.0, 1.0)
        )
