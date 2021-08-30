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

    def run_in_threaded(self, **kwargs):
        while True:
            self.agent.kwargs["point_cloud"] = self.run_in_series()

    def run_in_series(self) -> o3d.geometry.PointCloud:
        """

        :return: 3 x N array of point cloud
        """
        return self.pcd_via_open3d()

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
                                                                  depth_trunc=self.settings.depth_trunc)
        intric = self.agent.front_depth_camera.intrinsics_matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=rgb_data.shape[0],
                                                      height=rgb_data.shape[1],
                                                      fx=intric[0][0],
                                                      fy=intric[1][1],
                                                      cx=intric[0][2],
                                                      cy=intric[1][2]
                                                      )
        rot = self.agent.vehicle.transform.rotation
        loc = self.agent.vehicle.transform.location
        # xzy
        R = o3d.geometry.get_rotation_matrix_from_yzx(np.array(np.deg2rad([rot.pitch, rot.yaw, rot.roll])))
        T = np.array([loc.x, loc.y, loc.z])
        extrinsic = np.eye(4)
        extrinsic[0:3, 0:3] = R
        extrinsic[:3, 3] = T

        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd,
                                                                                      intrinsic=intrinsic,
                                                                                      extrinsic=extrinsic)

        # cords_y_minus_z_x = np.asarray(pcd.points)
        # cords_xyz_1 = np.vstack([
        #     cords_y_minus_z_x[:, 0],
        #     -cords_y_minus_z_x[:, 1],
        #     -cords_y_minus_z_x[:, 2],
        #     np.ones((1, np.shape(cords_y_minus_z_x)[0]))
        # ])
        # points = self.agent.vehicle.transform.get_matrix() @ cords_xyz_1
        # points = points.T[:, :3]
        # pcd.points = o3d.utility.Vector3dVector(points)

        if self.settings.should_down_sample:
            pcd = pcd.voxel_down_sample(self.settings.voxel_down_sample_size)
        return pcd

    def pcd_via_old_way(self):
        depth_img = self.agent.front_depth_camera.data.copy()
        coords = np.where(depth_img < self.settings.depth_trunc)  # it will just return all coordinate pairs
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if self.settings.should_down_sample:
            pcd = pcd.voxel_down_sample(self.settings.voxel_down_sample_size)
        pcd.paint_uniform_color(color=[0, 0, 0])
        return pcd

    def save(self, **kwargs):
        pass

    def _pix2xyz(self, depth_img, i, j):
        return [
            depth_img[i, j] * j * self.settings.depth_scale_raw,
            depth_img[i, j] * i * self.settings.depth_scale_raw,
            depth_img[i, j] * self.settings.depth_scale_raw
        ]


class DepthToPCDConfiguration(BaseModel):
    depth_scale_raw: float = Field(default=1, description="scaling of raw data")
    depth_trunc: float = Field(default=3, description="only depth less than this value will be taken")
    voxel_down_sample_size: float = Field(default=0.5, description="See open3d documetation on voxel downsample")
    should_down_sample: bool = Field(default=True, description="Whether to apply downsampling")
