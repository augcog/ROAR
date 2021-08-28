from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
import open3d as o3d


class iOSAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.points_added = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(iOSAgent, self).run_step(sensors_data, vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            depth_data = self.front_depth_camera.data.copy()
            rgb_data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))
            rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            rgb = o3d.geometry.Image(rgb_data)
            depth = o3d.geometry.Image(depth_data)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb, depth=depth,
                                                                      convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=rgb_data.shape[0],
                                                          height=rgb_data.shape[1],
                                                          fx=self.front_depth_camera.intrinsics_matrix[0][0],
                                                          fy=self.front_depth_camera.intrinsics_matrix[1][1],
                                                          cx=self.front_depth_camera.intrinsics_matrix[0][2],
                                                          cy=self.front_depth_camera.intrinsics_matrix[1][2]
                                                          )
            rot = self.vehicle.transform.rotation
            loc = self.vehicle.transform.location
            # o3d.geometry -> 180, 0, 180 == straight
            print(rot.pitch, rot.yaw, rot.roll)
            R = o3d.geometry.get_rotation_matrix_from_xyz(np.array(np.deg2rad([rot.pitch, rot.yaw, rot.roll])))
            T = np.array([loc.x, loc.y, loc.z])
            extrinsic = np.eye(4)
            extrinsic[0:3, 0:3] = R
            extrinsic[:3, 3] = T

            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd,
                                                                                          intrinsic=intrinsic,
                                                                                          extrinsic=extrinsic)
            # extrinsic=extrinsic)
            self.non_blocking_pcd_visualization(pcd=pcd)

        return VehicleControl()

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud, should_center=True):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()
