from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional
import time
from ROAR.utilities_module.utilities import img_to_world
import cv2
from numpy.matlib import repmat


class DepthToPointCloudDetector(Detector):
    def __init__(self,
                 agent: Agent,
                 should_compute_global_pointcloud: bool = False,
                 should_sample_points: bool = False,
                 should_filter_by_distance: float = False,
                 max_detectable_distance: float = 1,
                 scale_factor: int = 1000,
                 max_points_to_convert=10000, **kwargs):
        super().__init__(agent, **kwargs)
        self.should_compute_global_pointcloud = should_compute_global_pointcloud
        self.should_sample_points = should_sample_points
        self.should_filter_by_distance = should_filter_by_distance
        self.max_detectable_distance = max_detectable_distance
        self.max_points_to_convert = max_points_to_convert
        self.scale_factor = scale_factor

    def run_in_threaded(self, **kwargs):
        while True:
            self.agent.kwargs["point_cloud"] = self.run_in_series()

    def run_in_series(self) -> Optional[np.ndarray]:
        """

        :return: 3 x N array of point cloud
        """
        if self.agent.front_depth_camera.data is not None:
            # depth_img = self.agent.front_depth_camera.data.copy()
            # pixel_length = self.agent.front_depth_camera.image_size_x * self.agent.front_depth_camera.image_size_y
            # u_coord = repmat(np.r_[self.agent.front_depth_camera.image_size_x - 1:-1:-1],
            #                  self.agent.front_depth_camera.image_size_y, 1).reshape(pixel_length)
            # v_coord = repmat(np.c_[self.agent.front_depth_camera.image_size_y - 1:-1:-1],
            #                  1, self.agent.front_depth_camera.image_size_x).reshape(pixel_length)
            #
            # normalized_depth = np.reshape(depth_img, pixel_length)
            # p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
            # p3d = np.dot(np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix), p2d)
            # return p3d
            # p3d *= normalized_depth * 1000
            # return p3d
            depth_img = self.agent.front_depth_camera.data.copy()
            coords = np.where(depth_img < 10)  # it will just return all coordinate pairs
            raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]),
                                 (3, np.shape(coords)[1])).T  # N x 3

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
        return None

    @staticmethod
    def find_fps(t1, t2):
        return 1 / (t2 - t1)

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
