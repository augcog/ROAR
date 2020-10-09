from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional


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

    def run_step(self) -> Optional[np.ndarray]:
        """

        :return: 3 x N array of point cloud
        """
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data.copy()
            coords = np.where(depth_img < 2)
            # indices_to_select = np.random.choice(np.shape(coords)[1],
            #                                      size=min([self.max_points_to_convert, np.shape(coords)[1]]),
            #                                      replace=False)
            #
            # coords = (
            #     coords[0][indices_to_select],
            #     coords[1][indices_to_select]
            # )
            raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]),
                                 (3, np.shape(coords)[1])).T

            cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
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
                cords_xyz_1 = np.vstack([
                    cords_y_minus_z_x[2, :],
                    cords_y_minus_z_x[0, :],
                    -cords_y_minus_z_x[1, :],
                    np.ones((1, np.shape(cords_y_minus_z_x)[1]))
                ])
                points = self.agent.vehicle.transform.get_matrix() @ self.agent.front_depth_camera.transform.get_matrix() @ cords_xyz_1
                return points.T[:, :3]
        return None

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
