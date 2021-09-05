from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.rl_local_planner_1 import RLLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
import numpy as np
from typing import Any


class RLLocalPlannerAgent(Agent):
    def __init__(self, target_speed=40, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.logger = logging.getLogger("PID Agent")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = RLLocalPlanner(
            agent=self,
            controller=self.pid_controller)
        self.traditional_local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5
        )
        self.absolute_maximum_map_size, self.map_padding = 800, 40
        self.occupancy_map = OccupancyGridMap(absolute_maximum_map_size=800,
                                              world_coord_resolution=1,
                                              occu_prob=0.99,
                                              max_points_to_convert=10000,
                                              threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self,
                                                              threaded=True,
                                                              max_detectable_distance=0.5,
                                                              max_points_to_convert=20000,
                                                              min_obstacle_height=2)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        # self.add_threaded_module(self.occupancy_map)
        self.logger.debug(
            f"Waypoint Following Agent Initiated. Reading f"
            f"rom {self.route_file_path.as_posix()}")

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(RLLocalPlannerAgent, self).run_step(vehicle=vehicle,
                                                  sensors_data=sensors_data)
        self.traditional_local_planner.run_in_series()
        self.transform_history.append(self.vehicle.transform)
        if self.is_done:  # will never enter here
            control = VehicleControl()
            self.logger.debug("Path Following Agent is Done. Idling.")
        else:
            option = "obstacle_coords"  # ground_coords, point_cloud_obstacle_from_depth
            if self.kwargs.get(option, None) is not None:
                points = self.kwargs[option]
                self.occupancy_map.update(points)
            control = self.local_planner.run_in_series()
        return control