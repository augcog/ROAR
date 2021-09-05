from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.rl_local_planner import RLLocalPlanner
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
        self.absolute_maximum_map_size, self.map_padding = 1000, 40
        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self,threaded=True)
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
        option = "obstacle_coords"  # ground_coords, point_cloud_obstacle_from_depth
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update(points)
        control = self.local_planner.run_in_series()
        return control

    def get_obs(self):
        ch1 = self.occupancy_map.get_map(transform=self.vehicle.transform,
                                         view_size=(100, 100))
        ch1 = np.expand_dims((ch1 * 255).astype(np.uint8), -1)
        ch2 = np.zeros(shape=(100, 100, 1))
        ch3 = np.zeros(shape=ch2.shape)
        obs = np.concatenate([ch1, ch2, ch3], axis=2)
        print(np.shape(obs))
        return obs
