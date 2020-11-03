import logging
import open3d as o3d
import numpy as np
import cv2
from ROAR.perception_module.legacy.ground_plane_point_cloud_detector import \
    GroundPlanePointCloudDetector


class GPD_PC2(GroundPlanePointCloudDetector):
    def __init__(self,
                 max_ground_height_relative_to_vehcile=5,
                 knn=200,
                 std_ratio=2,
                 nb_neighbors=10,
                 **kwargs):
        """

        Args:
            max_detectable_distance: maximum detectable distance in km
            depth_scaling_factor: scaling depth back to world scale. 1000 m = 1 km
            **kwargs:
        """
        super().__init__(**kwargs)
        self.max_ground_height_relative_to_vehicle = max_ground_height_relative_to_vehcile
        self.logger = logging.getLogger("Point Cloud Detector")

    def run_in_series(self) -> np.ndarray:
        points_3d = self.calculate_world_cords()  # (Nx3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)  # - np.mean(points_3d, axis=0))
        pcd.estimate_normals()
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
        [k, idx, _] = pcd_tree.search_knn_vector_3d(
            self.agent.vehicle.transform.location.to_array(),
            knn=self.knn)  # find points around me
        points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3
        normals = np.asarray(pcd.normals)
        u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)
        ref_norm = vh[2, :]

        abs_diff = np.abs(normals @ ref_norm)  # N x 1
        plane_pixel_flat = 0.95 < abs_diff  # N x 1 (boolean)
        planes = points_3d[plane_pixel_flat]

        ref_z = self.agent.vehicle.transform.location.z
        ref_z = ref_z + self.max_ground_height_relative_to_vehicle
        ground_pixel_maybe = normals[:, 2] < ref_z  # N x 1 (boolean)
        plane_pixel_flat[ground_pixel_maybe] = False

        pcd.points = o3d.utility.Vector3dVector(points_3d[plane_pixel_flat])
        rgb_image = self.agent.front_rgb_camera.data.copy()  # W x H x 1
        d1, d2, _ = rgb_image.shape

        gpd_mask = plane_pixel_flat.reshape(d1, d2) # TODO the shape of plane_pixel_flat is 10000, but d1=600,d2=800, shape don't match
        rgb_image[gpd_mask, :] = 0

        cv2.imshow("Color", rgb_image)
        cv2.waitKey(1)

        """
        pcd, ids = pcd.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        points = np.asarray(pcd.points)
        self.pcd.points = o3d.utility.Vector3dVector(
                np.asarray(points_3d) - np.mean(points_3d, axis=0))
        """


        # if self.counter == 0:
        #     self.vis.create_window(window_name="Open3d", width=400, height=400)
        #     self.vis.add_geometry(self.pcd)
        #     render_option: o3d.visualization.RenderOption = self.vis.get_render_option()
        #     render_option.show_coordinate_frame = True
        # else:
        #     self.vis.update_geometry(self.pcd)
        #     render_option: o3d.visualization.RenderOption = self.vis.get_render_option()
        #     render_option.show_coordinate_frame = True
        #     self.vis.poll_events()
        #     self.vis.update_renderer()

        self.counter += 1
        return np.asarray(pcd.points)
