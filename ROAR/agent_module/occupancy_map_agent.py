from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
import cv2
import time

class OccupancyMapAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.add_threaded_module(DepthToPointCloudDetector(agent=self,
                                                           should_compute_global_pointcloud=True,
                                                           threaded=True,
                                                           scale_factor=1000))
        self.gpd = GroundPlaneDetector(agent=self, threaded=True)
        self.add_threaded_module(self.gpd)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        self.occupancy_map = OccupancyGridMap(scale=1)  # 1 m = 100 cm

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        control = self.local_planner.run_in_series()
        if self.kwargs.get("ground_coords") is not None:
            point_cloud: np.ndarray = self.kwargs.get("ground_coords")
            # print(np.amin(point_cloud, axis=0), np.amax(point_cloud, axis=0))
            t1 = time.time()
            print("ground", np.amin(point_cloud, axis=0), np.amax(point_cloud, axis=0),
                  self.vehicle.transform)
            self.occupancy_map.update(world_coords=point_cloud)

            # self.occupancy_map.vizualize(center=(self.vehicle.transform.location.x, self.vehicle.transform.location.y))
            # t2 = time.time()
            # print(1 / (t2 - t1))

        return control
