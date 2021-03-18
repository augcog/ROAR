from ROAR.perception_module.detector import Detector
import logging
import open3d as o3d
import numpy as np
import cv2
import time
from typing import Optional


class PointCloudDetector(Detector):
    def __init__(self, max_detectable_distance=0.05, depth_scaling_factor=1000, max_points_to_convert=10000, **kwargs):
        """

        Args:
            max_detectable_distance: maximum detectable distance in km
            depth_scaling_factor: scaling depth back to world scale. 1000 m = 1 km
            **kwargs:
        """
        super().__init__(**kwargs)
        self.max_detectable_distance = max_detectable_distance
        self.depth_scaling_factor = depth_scaling_factor
        self.max_points_to_convert = max_points_to_convert
        self.logger = logging.getLogger("Point Cloud Detector")
        self.pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()

        self.counter = 0

    def run_in_threaded(self, **kwargs):
        pass

    def run_in_series(self) -> Optional[np.ndarray]:
        points_3d = self.calculate_world_cords()  # (Nx3)
        return points_3d

    def calculate_world_cords(self):
        """Converts depth data from the Front Depth Camera to World coordinates."""
        depth_img = self.agent.front_depth_camera.data.copy()

        coords = np.where(depth_img < self.max_detectable_distance)

        indices_to_select = np.random.choice(np.shape(coords)[1],
                                             size=min([self.max_points_to_convert, np.shape(coords)[1]]),
                                             replace=False)

        coords = (
            coords[0][indices_to_select],
            coords[1][indices_to_select]
        ) # 600x800
        print(coords)
        raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]), (3, np.shape(coords)[1])).T

        cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
        cords_xyz_1 = np.vstack([
            cords_y_minus_z_x[0, :],
            -cords_y_minus_z_x[1, :],
            -cords_y_minus_z_x[2, :],
            np.ones((1, np.shape(cords_y_minus_z_x)[1]))
        ])

        points = self.agent.vehicle.transform.get_matrix() @ cords_xyz_1
        points = points.T[:, :3]
        return points

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
