from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.data_structures_models import Transform, Location
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.utilities_module.utilities import lengthSquare, getTriangleAngles
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
        self.current_region = 0
        self.set_mission_plan()
        self.logger.debug("Simple Path Following Local Planner Initiated")
        self.closeness_threshold = closeness_threshold
        
        if agent.agent_settings.waypoints_look_ahead_values: # ROAR Academy
            self.closeness_threshold_config = agent.agent_settings.waypoints_look_ahead_values
        else:
            self.closeness_threshold_config = json.load(Path(
            agent.agent_settings.simple_waypoint_local_planner_config_file_path).open(mode='r'))
                
    def set_mission_plan(self) -> None:
        """
        Clears current waypoints, and reset mission plan from start
        I am simply transferring the mission plan into my waypoint queue.
        Assuming that this current run will run all the way to the end

        current_region is also set to 0 as mission plan has been resetted to start.

        Returns:
            None
        """
        self.way_points_queue.clear()
        self.current_region = 0 # ROAR-Academy
        while self.mission_planner.mission_plan:  # this actually clears the mission plan!!
            self.way_points_queue.append(self.mission_planner.mission_plan.popleft())

    def is_done(self) -> bool:
        """
        If there are nothing in self.way_points_queue,
        that means you have finished a lap, you are done

        Returns:
            True if Done, False otherwise
        """
        return len(self.way_points_queue) == 0
    

    def update_current_region(self, vehicle_location:Location) -> int:
        '''
        Update current region
        '''
        while True:
            print("current_region:",self.current_region)
            left_waypoint = self.mission_planner.waypoints[self.current_region]
            right_waypoint = self.mission_planner.waypoints[self.current_region + 1]

            print("left_waypoint:", left_waypoint.location)
            print("right_waypoint:", right_waypoint.location)
            print("vehicle_location:", vehicle_location)

            alpha, betta, gamma = getTriangleAngles(
                A=(left_waypoint.location.x, left_waypoint.location.y), 
                B=(right_waypoint.location.x, right_waypoint.location.y),
                C=(vehicle_location.x, vehicle_location.y))
            if alpha > 90:
                self.current_region -= 1
                print("CR -= 1")
            elif betta < 90:
                self.current_region += 1
                print("CR += 1")
            else:
                print("CR Stays the Same")
                print("==========================================\n\n\n\n\n")
                return

    def run_in_series(self) -> VehicleControl:
        """
        Run step for the local planner
        Procedure:
            1. Sync data
            2. Update the current region # ROAR Academy
            3. get the correct look ahead for current speed for the current region
            4. get the correct next waypoint
            5. feed waypoint into controller
            6. return result from controller

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

        # update current region
        self.update_current_region(vehicle_transform.location)

        if vehicle_transform is None:
            raise AgentException("I do not know where I am, I cannot proceed forward")

        # redefine closeness level based on speed
        self.set_closeness_threhold(self.closeness_threshold_config) # ROAR-Academy: regional WLAV updtaed inside here. or right before this line.

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
        # self.logger.debug(f"\n"
        #                   f"Curr Transform: {self.agent.vehicle.transform}\n"
        #                   f"Target Location: {target_waypoint.location}\n"
        #                   f"Control: {control} | Speed: {Vehicle.get_speed(self.agent.vehicle)}\n")
        return control

    def set_closeness_threhold(self, config: dict):
        curr_speed = Vehicle.get_speed(self.agent.vehicle)
        for speed_upper_bound, closeness_threshold in config.items(): # ROAR-Academy: this needs to be updated to support regional setting.
            speed_upper_bound = float(speed_upper_bound)
            if curr_speed < speed_upper_bound:
                self.closeness_threshold = closeness_threshold
                break

    def restart(self):
        self.set_mission_plan()