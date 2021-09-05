import logging
import open3d as o3d
import numpy as np
from ROAR.perception_module.legacy.point_cloud_detector import PointCloudDetector


class GroundPlanePointCloudDetector(PointCloudDetector):
    def __init__(self,
                 max_ground_height_relative_to_vehcile=5,
                 knn=200,
                 std_ratio=2,
                 nb_neighbors=10,
                 ground_tilt_threshhold=0.05,
                 **kwargs):
        """

        Args:
            max_ground_height_relative_to_vehicle: anything above this height will be chucked away since it will be probaly ceiling
            knn: when finding reference points for ground detection, this is the number of points in front of the vehicle the algorithm will serach for
            std_ratio: this is the ratio that determines whether a point is an outlier. it is used in conjunction with nb_neighbor
            nb_neighbors: how many neighbors are around this point for it to be classified as "within" main frame
            ground_tilt_threshhold: variable to help compensate for slopes on the ground
            **kwargs:
        """
        super().__init__(**kwargs)
        self.logger = logging.getLogger("Point Cloud Detector")

        self.max_ground_height_relative_to_vehcile = max_ground_height_relative_to_vehcile
        self.knn = knn
        self.std_ratio = std_ratio
        self.nb_neighbors = nb_neighbors
        self.ground_tilt_threshold = ground_tilt_threshhold
        self.reference_normal = None

    def run_in_series(self) -> np.ndarray:
        points_3d, coords = self.calculate_world_cords()  # (Nx3) # TODO Christian, coords is a list of image X, Y that I've selected
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)  # - np.mean(points_3d, axis=0))
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        if self.reference_normal is None:
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
            [k, idx, _] = pcd_tree.search_knn_vector_3d(self.agent.vehicle.transform.location.to_array(), knn=self.knn)  # find points around me
            points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3
            u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)  # use svd to find normals of points
            self.reference_normal = vh[2, :]
        norm_flat = np.abs(normals @ self.reference_normal)
        planes = points_3d[norm_flat > 1-self.ground_tilt_threshold]
        ground = planes[planes[:, 2] < self.agent.vehicle.transform.location.z +
                        self.max_ground_height_relative_to_vehcile]
        return ground

    def calculate_world_cords(self):
        """Converts depth data from the Front Depth Camera to World coordinates."""
        depth_img = self.agent.front_depth_camera.data.copy()

        coords = np.where(depth_img < self.max_detectable_distance)

        indices_to_select = np.random.choice(np.shape(coords)[1],
                                             size=min([self.max_points_to_convert, np.shape(coords)[1]]),
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
        return points, coords
