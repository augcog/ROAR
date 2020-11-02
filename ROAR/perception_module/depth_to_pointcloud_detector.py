from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional
import time


class DepthToPointCloudDetector(Detector):
    def __init__(self,
                 agent: Agent,
                 should_compute_global_pointcloud: bool = False,
                 should_sample_points: bool = False,
                 should_filter_by_distance: float = False,
                 max_detectable_distance: float = 1,
                 max_points_to_convert=10000):
        super().__init__(agent)
        self.should_compute_global_pointcloud = should_compute_global_pointcloud
        self.should_sample_points = should_sample_points
        self.should_filter_by_distance = should_filter_by_distance
        self.max_detectable_distance = max_detectable_distance
        self.max_points_to_convert = max_points_to_convert

    def run_in_series(self) -> Optional[np.ndarray]:
        """

        :return: 3 x N array of point cloud
        """
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data.copy()
            if self.should_filter_by_distance:
                coords = np.where(depth_img < self.max_detectable_distance)
            else:
                coords = np.where(depth_img < 2)
            if self.should_sample_points and np.shape(coords)[1] > self.max_points_to_convert:
                coords = np.random.choice(a=coords, size=self.max_points_to_convert, replace=False)

            depths = depth_img[coords][:, np.newaxis] * 1000
            result = np.multiply(np.array(coords).T, depths)
            raw_p2d = np.hstack((result, depths))
            cords_xyz = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
            if self.should_compute_global_pointcloud:
                cords_xyz_1 = np.vstack([cords_xyz, np.ones((1, np.shape(cords_xyz)[1]))])
                return (self.agent.vehicle.transform.get_matrix() @ self.agent.front_depth_camera.transform.get_matrix() @ cords_xyz_1)[:3, :].T
            else:
                return cords_xyz.T
        return None

    @staticmethod
    def find_fps(t1, t2):
        return 1 / (t2 - t1)

