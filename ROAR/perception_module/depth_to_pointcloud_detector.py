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
        if "depth_image" in kwargs:
            return self.old_way(kwargs["depth_image"])
        return self.old_way(depth_img=self.agent.front_depth_camera.data.copy())
        # return self.pcd_via_open3d()

    def pcd_via_open3d(self):
        depth_data = self.agent.front_depth_camera.data.copy().astype(np.float32) * self.settings.depth_scale_raw
        rgb_data: np.ndarray = cv2.resize(self.agent.front_rgb_camera.data.copy(),
                                          dsize=(depth_data.shape[1], depth_data.shape[0]))

        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
        rgb = o3d.geometry.Image(rgb_data)
        depth = o3d.geometry.Image(depth_data)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb,
                                                                  depth=depth,
                                                                  convert_rgb_to_intensity=False,
                                                                  depth_scale=1,
                                                                  depth_trunc=100)
        intric = self.agent.front_depth_camera.intrinsics_matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=rgb_data.shape[0],
                                                      height=rgb_data.shape[1],
                                                      fx=intric[0][0],
                                                      fy=intric[1][1],
                                                      cx=intric[0][2],
                                                      cy=intric[1][2])
        extrinsics = self.agent.vehicle.transform.get_matrix()
        rot = self.agent.vehicle.transform.rotation
        extrinsics[0:3, 0:3] = o3d.geometry.get_rotation_matrix_from_xyz(rotation=
                                                                         np.deg2rad([rot.pitch, rot.yaw, rot.roll]))
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud. \
            create_from_rgbd_image(image=rgbd,
                                   intrinsic=intrinsic,
                                   extrinsic=extrinsics)
        if self.settings.should_down_sample:
            pcd = pcd.voxel_down_sample(self.settings.voxel_down_sample_size)
        return pcd

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
