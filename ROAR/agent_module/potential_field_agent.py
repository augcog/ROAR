from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.potential_field_planner import PotentialFieldPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.perception_module.obstacle_detector import ObstacleDetector
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from ROAR.perception_module.legacy.point_cloud_detector import PointCloudDetector
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
import time


class PotentialFieldAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)

        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)
        self.depth_to_obstacle = ObstacleFromDepth(agent=self, threaded=True)
        self.add_threaded_module(self.occupancy_map)
        self.add_threaded_module(self.depth_to_obstacle)
        # occu_map_file_path = Path("./ROAR_Sim/data/easy_map_cleaned_global_occu_map.npy")
        # self.occupancy_map.load_from_file(occu_map_file_path)


        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = PotentialFieldPlanner(agent=self,
                                                   behavior_planner=self.behavior_planner,
                                                   mission_planner=self.mission_planner,
                                                   controller=self.pid_controller)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.kwargs.get("obstacle_coords") is not None:
            self.occupancy_map.update_async(self.kwargs.get("obstacle_coords"))

        control = self.local_planner.run_in_series()
        return control
