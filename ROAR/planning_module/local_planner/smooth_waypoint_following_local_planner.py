from functools import reduce
from typing import Union

import numpy as np

from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.errors import (
    AgentException,
)
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl


class SmoothWaypointFollowingLocalPlanner(SimpleWaypointFollowingLocalPlanner):
    """
    Waypoint following local planner with waypoint lookahead for smoothing and target speed reduction.
    """

    def next_waypoint_smooth_and_speed(self, smooth_lookahead=400, speed_lookahead=600,
                                       min_speed_multiplier=0.6, speed_multiplier_slope=1.3) -> (Transform, float):
        """
        Calculate the next target waypoint and speed for the controller.

        Parameters
        smooth_lookahead : int
            Number of waypoints ahead to look at to compute the smoothed waypoint.
        speed_lookahead : int
            Number of waypoint to look ahaed to compute speed factor.
        min_speed_multiplier : float
            The minimum value for the speed multiplier.
        speed_multiplier_slope : float
            The rate of speed multiplier decrease for every 180 degrees of angle error.

        Returns
        target_waypoint : Transform
            The next target waypoint for the controller
        speed_multiplier : float
            The speed multiplier for the controller's target speed.
        """

        smooth_lookahead = min(smooth_lookahead, len(self.way_points_queue) - 1)
        speed_lookahead = min(speed_lookahead, len(self.way_points_queue) - 1)

        if smooth_lookahead > 10:  # Reduce computation by only looking at every 10 steps ahead
            sample_points = range(0, smooth_lookahead, smooth_lookahead // 10)
            location_sum = reduce(lambda x, y: x + y,
                                  (self.way_points_queue[i].location for i in sample_points))
            rotation_sum = reduce(lambda x, y: x + y,
                                  (self.way_points_queue[i].rotation for i in sample_points))

            num_points = len(sample_points)
            target_waypoint = Transform(location=location_sum / num_points, rotation=rotation_sum / num_points)
        else:
            target_waypoint = self.way_points_queue[-1]

        if speed_lookahead > 0:
            angle_difference = self._calculate_angle_error(self.way_points_queue[speed_lookahead])
            # Angle difference is between 0 and 180, but unlikely to be more than 90
            speed_multiplier = max(min_speed_multiplier,
                                   (1.0 - speed_multiplier_slope * angle_difference / np.pi))
        else:
            speed_multiplier = 1.0

        return target_waypoint, speed_multiplier

    def run_in_series(self) -> VehicleControl:
        """
        Run step for the local planner
        Procedure:
            1. Sync data
            2. get the correct look ahead for current speed
            3. get the correct next waypoint
            4. feed waypoint into controller
            5. return result from controller

        Returns:
            next control that the local think the agent should execute.
        """
        if (
                len(self.mission_planner.mission_plan) == 0
                and len(self.way_points_queue) == 0
        ):
            return VehicleControl()

        # get vehicle's location
        vehicle_transform: Union[Transform, None] = self.agent.vehicle.control

        if vehicle_transform is None:
            raise AgentException("I do not know where I am, I cannot proceed forward")

        # redefine closeness level based on speed
        self.set_closeness_threhold(self.closeness_threshold_config)

        # get current waypoint
        curr_closest_dist = float("inf")
        while True:
            if len(self.way_points_queue) == 0:
                self.logger.info("Destination reached")
                return VehicleControl()
            # waypoint: Transform = self.way_points_queue[0]
            waypoint, speed_factor = self.next_waypoint_smooth_and_speed()
            curr_dist = vehicle_transform.location.distance(waypoint.location)
            if curr_dist < curr_closest_dist:
                # if i find a waypoint that is closer to me than before
                # note that i will always enter here to start the calculation for curr_closest_dist
                curr_closest_dist = curr_dist
            elif curr_dist < self.closeness_threshold:
                # i have moved onto a waypoint, remove that waypoint from the queue
                self.way_points_queue.popleft()
            else:
                break

        target_waypoint, speed_factor = self.next_waypoint_smooth_and_speed()
        control: VehicleControl = self.controller.run_in_series(next_waypoint=target_waypoint,
                                                                speed_multiplier=speed_factor)
        self.logger.debug(f"\n"
                          f"Curr Transform: {self.agent.vehicle.control}\n"
                          f"Target Location: {target_waypoint.location}\n"
                          f"Control: {control} | Speed: {Vehicle.get_speed(self.agent.vehicle)}\n")
        return control

    def _calculate_angle_error(self, next_waypoint: Transform):
        """
        Compute the angle difference between the current vehicle position and orientation and the specified waypoint.
        Ignore the height component (y).

        Parameters
        next_waypoint : Transform
            The waypoint to compute the angle against.

        Returns
        : float
            The angle difference in radians in the 0 to pi range.
        """

        v_vec = np.array([np.cos(np.radians(self.agent.vehicle.control.rotation.pitch)),
                          0,
                          np.sin(np.radians(self.agent.vehicle.control.rotation.pitch))])
        v_vec_norm = np.linalg.norm(v_vec)  # Already norm 1

        w_vec = next_waypoint.location.to_array() - self.controller.agent.vehicle.control.location.to_array()
        w_vec[1] = 0.
        w_vec_norm = np.linalg.norm(w_vec)

        return np.math.acos(np.dot(v_vec, w_vec) / (v_vec_norm * w_vec_norm))  # 0 to np.pi

