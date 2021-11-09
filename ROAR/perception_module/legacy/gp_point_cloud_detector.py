from typing import Any
from ROAR.perception_module.legacy.ground_plane_point_cloud_detector import GroundPlanePointCloudDetector
import cv2
import numpy as np
import open3d as o3d


class GP_pointcloud_detector(GroundPlanePointCloudDetector):
    def run_in_series(self) -> Any:
        super(GP_pointcloud_detector, self).run_in_series()
        if self.agent.front_depth_camera.data is not None:
            depth_image: np.ndarray = self.agent.front_depth_camera.data.copy()
            color_image = self.agent.front_rgb_camera.data.copy()

            # transfer to world coordinate points
            points_3d = self.calculate_world_cords()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)  # - np.mean(points_3d, axis=0))
            pcd.estimate_normals()
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
            [k, idx, _] = pcd_tree.search_knn_vector_3d(self.agent.vehicle.transform.location.to_array(),
                                                        knn=200)  # find points around me
            points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3

            u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)  # use svd to find normals of points

            # find norm for points near by
            avg_points_near_me_normal = vh[2, :]

            # find norm for all points
            normals = np.asarray(pcd.normals)

            # find all 3d points that belong to ground
            abs_diff = np.linalg.norm(normals - avg_points_near_me_normal, axis=1)  # anything below avg is plane
            planes = points_3d[abs_diff < np.mean(abs_diff)]
            ground = planes[planes[:, 2] < self.agent.vehicle.transform.location.z + 0.9]

            # project points back to 2d
            img_coords: np.ndarray = self.world_to_img_transform(ground)[:,:2]

            color_image[img_coords[:, 1], img_coords[:, 0]] = [255, 255, 255]
            #
            cv2.imshow("rgb", color_image)
            cv2.waitKey(1)

    def calculate_world_cords(self):
        """Converts depth data from the Front Depth Camera to World coordinates."""
        depth_img = self.agent.front_depth_camera.data.copy()

        coords = np.where(depth_img < 0.9)

        indices_to_select = np.random.choice(np.shape(coords)[1],
                                             size=min([10000, np.shape(coords)[1]]),
                                             replace=False)

        coords = (
            coords[0][indices_to_select],
            coords[1][indices_to_select]
        )

        raw_p2d = np.reshape(self._pix2xyz(depth_img=depth_img, i=coords[0], j=coords[1]), (3, np.shape(coords)[1])).T

        cords_y_minus_z_x = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
        cords_xyz_1 = np.vstack([
            cords_y_minus_z_x[2, :],
            cords_y_minus_z_x[0, :],
            -cords_y_minus_z_x[1, :],
            np.ones((1, np.shape(cords_y_minus_z_x)[1]))
        ])
        points: np.ndarray = self.agent.vehicle.transform.get_matrix() @ self.agent.front_depth_camera.transform.get_matrix() @ cords_xyz_1
        points = points.T[:, :3]
        return points

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]

    def world_to_img_transform(self, xyz: np.ndarray) -> np.ndarray:
        """
        Calculate the 2D image coordinate from 3D world space

        Args:
            xyz: (Nx3) array representing X, Y, Z in world coord

        Returns:
            Array if integers [u, v, f]

        """
        xyz1 = np.append(xyz, np.ones(shape=(len(xyz), 1)), axis=1)
        veh_cam_matrix = self.agent.front_depth_camera.transform.get_matrix()  # 4 x 4
        world_veh_matrix = self.agent.vehicle.transform.get_matrix()  # 4 x 4

        world_cam_matrix = np.linalg.inv(np.dot(world_veh_matrix, veh_cam_matrix))
        cords_xyz1 = world_cam_matrix @ xyz1.T
        cords_y_minus_z_x = np.array([cords_xyz1[1, :], -cords_xyz1[2, :], cords_xyz1[0, :]])
        raw_p2d = self.agent.front_depth_camera.intrinsics_matrix @ cords_y_minus_z_x
        cam_cords = np.array(
            [raw_p2d[0, :] / raw_p2d[2, :], raw_p2d[1, :] / raw_p2d[2, :], raw_p2d[2, :]]
        ).T
        return np.round(cam_cords, 0).astype(np.int64)