from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.utilities_module.data_structures_models import Transform
import numpy as np
import math
from ROAR.agent_module.agent import Agent

"""
Citation: https://github.com/AtsushiSakai/PythonRobotics/blob/master
/PathTracking/pure_pursuit/pure_pursuit.py
"""


class PurePursuitController(Controller):
    def __init__(
            self,
            agent: Agent,
            look_ahead_gain: float = 0.1,
            look_ahead_distance: float = 2,
            target_speed=60,
    ):
        """

        Args:
            vehicle: Vehicle information
            look_ahead_gain: Look ahead factor
            look_ahead_distance: look ahead distance
            target_speed: desired longitudinal speed to maintain
        """

        super(PurePursuitController, self).__init__(agent=agent)
        self.target_speed = self.agent.agent_settings.max_speed \
            if self.agent.agent_settings.max_speed else target_speed
        self.look_ahead_gain = look_ahead_gain
        self.look_ahead_distance = look_ahead_distance
        self.latitunal_controller = LatitunalPurePursuitController(
            agent=self.agent,
            look_ahead_gain=look_ahead_gain,
            look_ahead_distance=look_ahead_distance,
        )
        self.longitunal_controller = LongitunalPurePursuitController(
            agent=self.agent, target_speed=target_speed
        )

    def run_in_series(
            self, next_waypoint: Transform, **kwargs
    ) -> VehicleControl:
        """
        run one step of Pure Pursuit Control

        Args:
            vehicle: current vehicle state
            next_waypoint: Next waypoint, Transform
            **kwargs:

        Returns:
            Vehicle Control

        """
        control = VehicleControl(
            throttle=self.longitunal_controller.run_step(),
            steering=self.latitunal_controller.run_step(next_waypoint=next_waypoint),
        )
        return control


class LongitunalPurePursuitController:
    def __init__(self, agent: Agent, target_speed=60, kp=0.1):
        self.agent = agent
        self.target_speed = target_speed
        self.kp = kp

    def run_step(self) -> float:
        return float(
            VehicleControl.clamp(
                self.kp * (self.target_speed - Vehicle.get_speed(self.agent.vehicle)), 0,
                1
            )
        )


class LatitunalPurePursuitController:
    def __init__(
            self, agent: Agent, look_ahead_gain: float,
            look_ahead_distance: float
    ):
        self.agent = agent
        self.look_ahead_gain = look_ahead_gain
        self.look_ahead_distance = look_ahead_distance

    def run_step(self, next_waypoint: Transform) -> float:
        target_z = next_waypoint.location.z
        target_x = next_waypoint.location.x
        angle_difference = math.atan2(
            target_z - self.agent.vehicle.transform.location.z,
            target_x - self.agent.vehicle.transform.location.x
        ) - np.radians(self.agent.vehicle.transform.rotation.pitch)
        curr_look_forward = (
                self.look_ahead_gain * Vehicle.get_speed(vehicle=self.agent.vehicle)
                + self.look_ahead_distance
        )
        lateral_difference = math.atan2(
            2.0 * self.agent.vehicle.wheel_base * math.sin(angle_difference) / curr_look_forward,
            1.0,
        )
        return VehicleControl.clamp(lateral_difference, -1, 1)
