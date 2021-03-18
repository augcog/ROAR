from typing import Any

from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
import open3d


class ObstacleFromDepth(Detector):
    def __init__(self, agent: Agent,
                 max_detectable_distance: float = 0.3,
                 max_points_to_convert: int = 10000,
                 max_incline_normal=0.5,
                 min_obstacle_height: float = 3, ** kwargs):

        super().__init__(agent, **kwargs)
        self.max_detectable_distance = kwargs.get("max_detectable_distance", max_detectable_distance)
        self.max_points_to_convert = kwargs.get("max_points_to_convert", max_points_to_convert)
        self.max_incline_normal = kwargs.get("max_incline_normal", max_incline_normal)
        self.min_obstacle_height = kwargs.get("max_obstacle_height", min_obstacle_height)

    def run_in_series(self, **kwargs) -> Any:
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data.copy()
            coords = np.where(depth_img < self.max_detectable_distance)

            indices_to_select = np.random.choice(np.shape(coords)[1],
                                                 size=min([self.max_points_to_convert, np.shape(coords)[1]]),
                                                 replace=False)

            coords = (
                coords[0][indices_to_select],
                coords[1][indices_to_select]
            )
            raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]),
                                 (3, np.shape(coords)[1])).T

            cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
            cords_xyz_1 = np.vstack([
                cords_y_minus_z_x[0, :],
                -cords_y_minus_z_x[1, :],
                -cords_y_minus_z_x[2, :],
                np.ones((1, np.shape(cords_y_minus_z_x)[1]))
            ])

            points = self.agent.vehicle.transform.get_matrix() @ cords_xyz_1
            points = points.T[:, :3]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            pcd.normalize_normals()

            normals = np.asarray(pcd.normals)
            abs_normals = np.abs(normals)
            obstacles_mask = abs_normals[:, 1] < self.max_incline_normal
            obstacle_below_height_mask = points[:, 1] < self.agent.vehicle.transform.location.y + self.min_obstacle_height
            mask = obstacles_mask & obstacle_below_height_mask
            self.agent.kwargs["point_cloud_obstacle_from_depth"] = points
            self.agent.kwargs["obstacle_coords"] = points[mask]
            self.agent.kwargs["ground_coords"] = points[~mask]
            return self.agent.kwargs["obstacle_coords"]

    def run_in_threaded(self, **kwargs):
        while True:
            self.run_in_series()

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
