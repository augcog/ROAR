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
            depth_img = self.agent.front_depth_camera.data.copy()
            coords = np.where(depth_img <= np.amax(depth_img))  # it will just return all coordinate pairs
            depths = depth_img[coords][:, np.newaxis] * self.scale_factor
            result = np.multiply(np.array(coords).T, depths)
            S_uv1 = np.hstack((result, depths)).T
            if self.should_compute_global_pointcloud:
                result = img_to_world(scaled_depth_image=S_uv1,
                                      intrinsics_matrix=self.agent.front_depth_camera.intrinsics_matrix,
                                      veh_world_matrix=self.agent.vehicle.transform.get_matrix(),
                                      cam_veh_matrix=self.agent.front_depth_camera.transform.get_matrix())
                return result

            else:
                # return p3d.T
                K_inv = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix)
                return (K_inv @ S_uv1).T
        return None


    @staticmethod
    def find_fps(t1, t2):
        return 1 / (t2 - t1)
