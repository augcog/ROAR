from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.control_module.controller import Controller
from ROAR.planning_module.mission_planner.mission_planner import MissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner

import logging
from typing import Union
from ROAR.utilities_module.errors import (
    AgentException,
)
from ROAR.agent_module.agent import Agent
import json
from pathlib import Path


class SimpleWaypointFollowingLocalPlanner(LocalPlanner):
    def __init__(
            self,
            agent: Agent,
            controller: Controller,
            mission_planner: MissionPlanner,
            behavior_planner: BehaviorPlanner,
            closeness_threshold=0.5,
    ):
        """
        Initialize Simple Waypoint Following Planner
        Args:
            agent: newest agent state
            controller: Control module used
            mission_planner: mission planner used
            behavior_planner: behavior planner used
            closeness_threshold: how close can a waypoint be with the vehicle
        """
        super().__init__(agent=agent,
                         controller=controller,
                         mission_planner=mission_planner,
                         behavior_planner=behavior_planner,
                         )
        self.logger = logging.getLogger("SimplePathFollowingLocalPlanner")
        self.set_mission_plan()
        self.logger.debug("Simple Path Following Local Planner Initiated")
        self.closeness_threshold = closeness_threshold
        self.closeness_threshold_config = json.load(Path(
            agent.agent_settings.simple_waypoint_local_planner_config_file_path).open(mode='r'))

    def set_mission_plan(self) -> None:
        """
        clears current waypoints, and reset mission plan from start
        I am simply transfering the mission plan into my waypoint queue.
        Assuming that this current run will run all the way to the end

        Returns:
            None
        """
        self.way_points_queue.clear()
        while (
                self.mission_planner.mission_plan
        ):  # this actually clears the mission plan!!
            self.way_points_queue.append(self.mission_planner.mission_plan.popleft())

    def is_done(self) -> bool:
        """
        If there are nothing in self.way_points_queue,
        that means you have finished a lap, you are done

        Returns:
            True if Done, False otherwise
        """
        return len(self.way_points_queue) == 0

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
        vehicle_transform: Union[Transform, None] = self.agent.vehicle.transform

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
            waypoint: Transform = self.way_points_queue[0]
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

        target_waypoint = self.way_points_queue[0]
        control: VehicleControl = self.controller.run_in_series(next_waypoint=target_waypoint)
        self.logger.debug(f"\n"
                          f"Curr Transform: {self.agent.vehicle.transform}\n"
                          f"Target Location: {target_waypoint.location}\n"
                          f"Control: {control} | Speed: {Vehicle.get_speed(self.agent.vehicle)}\n")
        return control

    def set_closeness_threhold(self, config: dict):
        curr_speed = Vehicle.get_speed(self.agent.vehicle)
        for speed_upper_bound, closeness_threshold in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if curr_speed < speed_upper_bound:
                self.closeness_threshold = closeness_threshold
                break
