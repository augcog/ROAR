from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional
import time


class DepthToPointCloudDetector(Detector):
    def __init__(self,
                 agent: Agent,
                 compute_global_pointcloud: bool = False,
                 max_detectable_distance: float = 0.05,
                 max_points_to_convert=10000):
        super().__init__(agent)
        self.compute_global_pointcloud = compute_global_pointcloud
        self.max_detectable_distance = max_detectable_distance
        self.max_points_to_convert = max_points_to_convert

    def run_in_series(self) -> Optional[np.ndarray]:
        """

        :return: 3 x N array of point cloud
        """
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data.copy()
            t1 = time.time()
            coords = np.where(depth_img < 2)
            t2 = time.time()
            # raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]),
            #                      (3, np.shape(coords)[1])).T
            depths = depth_img[coords][:, np.newaxis] * 1000
            result = np.multiply(np.array(coords).T, depths)
            raw_p2d = np.hstack((result, depths))
            cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T

            t4 = time.time()
            if self.compute_global_pointcloud:
                cords_xyz_1 = np.vstack([
                    cords_y_minus_z_x[2, :],
                    cords_y_minus_z_x[0, :],
                    -cords_y_minus_z_x[1, :],
                    np.ones((1, np.shape(cords_y_minus_z_x)[1]))
                ])
                points: np.ndarray = self.agent.vehicle.transform.get_matrix() @ self.agent.front_depth_camera.transform.get_matrix() @ cords_xyz_1
                points = points.T[:, :3]
                return points
            else:
                points = np.vstack([
                    cords_y_minus_z_x[2, :],
                    cords_y_minus_z_x[0, :],
                    -cords_y_minus_z_x[1, :]])
                t5 = time.time()
                print(f"FPS to find all valid points = {self.find_fps(t1, t2)} | "
                      f"FPS to convert image to 3D points = {self.find_fps(t2, t4)} | "
                      f"FPS to transform to world/relative coords = {self.find_fps(t4, t5)}")
                return points.T
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
