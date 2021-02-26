from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl

from ROAR.perception_module.legacy.ground_plane_point_cloud_detector import GroundPlanePointCloudDetector
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap


class PointCloudAgent(Agent):
    def __init__(self, **kwargs):
        super(PointCloudAgent, self).__init__(**kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.controller = PurePursuitController(agent=self, target_speed=20)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        """
        self.gp_pointcloud_detector = GroundPlanePointCloudDetector(agent=self,
                                                                    max_points_to_convert=10000,
                                                                    nb_neighbors=100,
                                                                    std_ratio=1)
        """
        self.gp_pointcloud_detector = GroundPlanePointCloudDetector(agent=self,
                                                                    max_points_to_convert=10000,
                                                                    nb_neighbors=100,
                                                                    std_ratio=1)

        self.occupancy_grid_map = OccupancyGridMap(absolute_maximum_map_size=800)
        # self.visualizer = Visualizer(agent=self)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(PointCloudAgent, self).run_step(sensors_data, vehicle)
        try:

            self.local_planner.run_in_series()
            points = self.gp_pointcloud_detector.run_in_series()  # (N x 3)
            self.occupancy_grid_map._update_grid_map_from_world_cord(world_cords_xy=points[:, :2])
            self.occupancy_grid_map.visualize(vehicle_location=self.vehicle.transform.location)
            # print(np.amin(points, axis=0), np.amax(points, axis=0), self.vehicle.transform.location.to_array())
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # o3d.visualization.draw_geometries([pcd])
            # self.occupancy_grid_map.update_grid_map_from_world_cord(points[:, :2])
            # self.occupancy_grid_map.visualize(vehicle_location=self.vehicle.transform.location)

        except Exception as e:
            self.logger.error(f"Point cloud RunStep Error: {e}")
        finally:
            return self.local_planner.run_in_series()
