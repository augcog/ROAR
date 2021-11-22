from ROAR.agent_module.agent import Agent
from ROAR.agent_module.line_following_agent_2 import LineFollowingAgent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
import open3d as o3d
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.lane_detector import LaneDetector
from scipy.spatial.transform import Rotation as R


class OccupancyMapAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        # initialize occupancy grid map content
        self.occu_map = OccupancyGridMap(agent=self)
        self.depth_to_pcd = DepthToPointCloudDetector(agent=self)
        self.ground_plane_detector = GroundPlaneDetector(agent=self)
        # initialize open3d related content
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.points_added = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(OccupancyMapAgent, self).run_step(sensors_data, vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            depth_img = self.front_depth_camera.data.copy()
            pcd = self.depth_to_pcd.run_in_series(depth_image=depth_img)
            self.non_blocking_pcd_visualization(pcd=pcd, should_center=True,
                                                should_show_axis=True, axis_size=1)
        return self.vehicle.control

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud,
                                       should_center=False,
                                       should_show_axis=False,
                                       axis_size: float = 0.1):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                          origin=np.mean(points,
                                                                                                         axis=0))
                self.vis.add_geometry(self.coordinate_frame)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                          origin=np.mean(points,
                                                                                                         axis=0))
                self.vis.update_geometry(self.coordinate_frame)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
