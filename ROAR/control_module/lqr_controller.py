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
from scipy.linalg import solve_discrete_are as dare


# written by Alvin Tan during Fall 2020 for his EECS 206A final project at UC Berkeley
class LQRController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        
        # load in system matrices
        self.config = json.load(Path(agent.agent_settings.lqr_config_file_path).open(mode='r'))
        self.A = np.array(self.config['A'])
        self.B = np.array(self.config['B'])
        self.Q = np.array(self.config['Q'])
        self.R = np.array(self.config['R'])
        # calculate our feedback matrix
        self.P, self.K = self._dlqr(self.A,
                                    self.B,
                                    self.Q,
                                    self.R)

        # some reactive speed control stuff
        self.errBoi = 0
        self.errAlpha = self.config['errAlpha']
        self.slowdown = self.config['slowdown']
        self.maxSlow = self.config['maxSlow']

        self.logger = logging.getLogger(__name__)

    # solves the infinite-horizon discrete-time lqr 
    def _dlqr(self, A, B, Q, R):
        # solve the ricatti equation for P
        P = dare(A, B, Q, R)
        
        # solve for our feedback matrix 
        # K = (B.T P B + R)^-1 (B.T P A)
        K = np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([B.T, P, B]) + R), B.T, P, A])
        
        return P, K

    def run_in_series(self, next_waypoint: Transform, speed_multiplier=1.0, **kwargs) -> VehicleControl:
        # Calculate the current angle to the next waypoint
        angBoi = -self._calculate_angle_error(next_waypoint=next_waypoint)
        # Grab our current speed
        curSpeed = Vehicle.get_speed(self.agent.vehicle)
        # Toss both values into a current xt
        xt = np.array([angBoi, curSpeed])
        
        # Generate our target speed with reactive speed reduction when off track
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed)) * speed_multiplier
        # if we are very off track, update error to reflect that
        absErr = np.abs(angBoi)
        if absErr > self.errBoi:
            self.errBoi = absErr
        else: # if we are getting back on track, gradually reduce our error 
            self.errBoi = self.errBoi*(1-self.errAlpha) + absErr*self.errAlpha
        # reduce our target speed based on how far off target we are
        # target_speed *= (math.exp(-self.errBoi) - 1) * self.slowdown + 1
        target_speed *= max((math.cos(self.errBoi) - 1) * self.slowdown, -self.maxSlow) + 1

        ## Note for future: It may be helpful to have another module for adaptive speed control and some way to slowly
        ## increase the target speed when we can.
        
        # Assume we want to go in the direction of the waypoint at the target speed foreversies
        xd = np.array([0, target_speed])
        cur_speed = Vehicle.get_speed(self.agent.vehicle)
        cd = np.array([0, cur_speed])
        # Calculate the feasible ud trajectory
        ud,_,_,_ = np.linalg.lstsq(self.B, xd-np.dot(self.A, cd), rcond=None)
        
        # convert to offset variables zt and ht
        zt = xt - xd
        ht = -np.dot(self.K, zt)
        # convert back to ut and clip our inputs
        ut = ht + ud
        steering = np.clip(ut[0], self.steering_boundary[0], self.steering_boundary[1])
        throttle = np.clip(ut[1], self.throttle_boundary[0], self.throttle_boundary[1])
        
        return VehicleControl(steering=steering, throttle=throttle)
        
    # code stolen from the PID controller to calculate the angle
    def _calculate_angle_error(self, next_waypoint: Transform):
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.control.location
        v_end = v_begin + Location(
            x=math.cos(math.radians(self.agent.vehicle.control.rotation.pitch)),
            y=v_begin.y,
            z=math.sin(math.radians(self.agent.vehicle.control.rotation.pitch)),
        )
        # we ignore the vertical/altitude component, which is y, and only consider horizontal angle for
        # steering control
        v_vec = np.array([v_end.x - v_begin.x, 0, v_end.z - v_begin.z])

        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin.x,
                0, # again, ignoring vertical component
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

