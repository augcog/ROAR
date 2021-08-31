from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from pydantic import BaseModel, Field
import cv2
import open3d as o3d


class DepthToPointCloudDetector(Detector):
    def __init__(self,
                 agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.settings: DepthToPCDConfiguration = DepthToPCDConfiguration.parse_file(
            self.agent.agent_settings.depth_to_pcd_config_path)
        self.should_take_only_red = False
        self.use_floodfill = False

    def run_in_threaded(self, **kwargs):
        while True:
            self.agent.kwargs["point_cloud"] = self.run_in_series()

    def run_in_series(self) -> o3d.geometry.PointCloud:
        """

        :return: 3 x N array of point cloud
        """

        return self.old_way()

    def old_way(self):
        depth_img = self.agent.front_depth_camera.data.copy()
        coords = np.where(depth_img < self.settings.depth_trunc)  # it will just return all coordinate pairs
        Is = coords[0][::self.settings.depth_image_sample_step_size]
        Js = coords[1][::self.settings.depth_image_sample_step_size]
        raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=Is, j=Js),
                             (3, len(Is))).T  # N x 3
        intrinsic = self.agent.front_depth_camera.intrinsics_matrix
        cords_y_minus_z_x: np.ndarray = np.linalg.inv(intrinsic) @ raw_p2d.T
        cords_xyz_1 = np.vstack((cords_y_minus_z_x, np.ones((1, cords_y_minus_z_x.shape[1]))))
        points = self.agent.vehicle.transform.get_matrix() @ cords_xyz_1
        points = points.T[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if self.settings.should_down_sample:
            pcd = pcd.voxel_down_sample(self.settings.voxel_down_sample_size)
        pcd.paint_uniform_color(color=[0, 0, 0])
        return pcd

    def save(self, **kwargs):
        pass

    def _pix2xyz(self, depth_img, i, j):
        return np.array([depth_img[i, j] * j,
                         depth_img[i, j] * i,
                         depth_img[i, j]
                         ]) * self.settings.depth_scale_raw


class DepthToPCDConfiguration(BaseModel):
    depth_scale_raw: float = Field(default=1, description="scaling of raw data")
    depth_trunc: float = Field(default=3, description="only depth less than this value will be taken")
    voxel_down_sample_size: float = Field(default=0.5, description="See open3d documetation on voxel downsample")
    should_down_sample: bool = Field(default=True, description="Whether to apply downsampling")
    depth_image_sample_step_size: int = Field(default=10, description="Step size for sampling depth image")
