from typing import Any

from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
import open3d
from pydantic import BaseModel, Field


class ObstacleFromDepth(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)
        config = ObstacleFromDepthConfig.parse_file(self.agent.agent_settings.obstacle_from_depth_config_path)
        self.max_detectable_distance = kwargs.get("max_detectable_distance", config.max_detectable_distance)
        self.max_points_to_convert = kwargs.get("max_points_to_convert", config.max_points_to_convert)
        self.max_incline_normal = kwargs.get("max_incline_normal", config.max_incline_normal)
        self.min_obstacle_height = kwargs.get("max_obstacle_height", config.min_obstacle_height)

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
            obstacle_below_height_mask = \
                np.abs(points[:, 1]) < self.agent.vehicle.transform.location.y + self.min_obstacle_height
            mask = obstacles_mask & obstacle_below_height_mask
            self.agent.kwargs["point_cloud_obstacle_from_depth"] = points
            self.agent.kwargs["obstacle_coords"] = points[mask]
            self.agent.kwargs["ground_coords"] = points[~mask]
            return self.agent.kwargs["obstacle_coords"]

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]


class ObstacleFromDepthConfig(BaseModel):
    max_detectable_distance: float = Field(default=0.3)
    max_points_to_convert: int = Field(default=10000)
    max_incline_normal: float = Field(default=0.5)
    min_obstacle_height: float = Field(default=3)
    update_interval: float = Field(default=0.1)
