from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional, Any
import open3d as o3d
import time, cv2


class GroundPlaneDetector(Detector):
    def __init__(self, agent: Agent, knn: int = 200, **kwargs):
        super().__init__(agent, **kwargs)
        self.reference_norm: Optional[np.ndarray] = np.array([-0.00000283, -0.00012446, 0.99999999])
        self.knn = knn
        self.f1, self.f2, self.f3, self.f4 = self.compute_vectors_near_me()

    def run_in_threaded(self, **kwargs):
        while True:
            self.run_in_series()

    def run_in_series(self) -> Any:
        if self.agent.kwargs.get("point_cloud", None) is not None:
            points: np.ndarray = self.agent.kwargs.get("point_cloud")
            x = points[self.f3, :] - points[self.f4, :]
            y = points[self.f1, :] - points[self.f2, :]
            normals = self.normalize_v3(np.cross(x, y))
            # OpenCV FloodFill
            d1 = self.agent.front_depth_camera.image_size_y
            d2 = self.agent.front_depth_camera.image_size_x
            new_d1, new_d2 = d1, d2
            curr_img = normals.reshape((new_d1, new_d2, 3)).astype(np.float32)
            seed_point = (new_d1 * 2 // 3, new_d2 // 2)
            diff = 0.01
            diffs = (diff, diff, diff)
            _, retval, _, _ = cv2.floodFill(image=curr_img,
                                            seedPoint=seed_point,
                                            newVal=(0, 0, 0),
                                            loDiff=diffs,
                                            upDiff=diffs,
                                            mask=None)
            bool_matrix = np.mean(retval, axis=2) == 0
            bool_zeros = np.zeros(d1 * d2).flatten()
            bool_indices = np.indices(bool_zeros.shape)[0]  # [::16]
            bool_zeros[bool_indices] = bool_matrix.flatten()
            bool_matrix = bool_zeros.reshape((d1, d2))
            # color_image = self.agent.front_rgb_camera.data.copy()
            # color_image[bool_matrix > 0] = 255
            # cv2.imshow('Color', color_image)
            # cv2.waitKey(1)
            ground_coords = np.where(bool_matrix > 0)
            self.agent.kwargs["ground_coords"] = self.to_world_coords(img_ground_coords=ground_coords)
        # time1 = time.time()
        # points = super(GroundPlaneDetector, self).run_in_series()  # Nx3
        # time2 = time.time()
        # x = points[self.f3, :] - points[self.f4, :]
        # y = points[self.f1, :] - points[self.f2, :]
        # normals = self.normalize_v3(np.cross(x, y))
        # time3 = time.time()
        # # OpenCV FloodFill
        # d1 = self.agent.front_depth_camera.image_size_y
        # d2 = self.agent.front_depth_camera.image_size_x
        # new_d1, new_d2 = d1, d2
        # curr_img = normals.reshape((new_d1, new_d2, 3)).astype(np.float32)
        # seed_point = (new_d1 * 2 // 3, new_d2 // 2)
        # _, retval, _, _ = cv2.floodFill(image=curr_img,
        #                                 seedPoint=seed_point,
        #                                 newVal=(0, 0, 0),
        #                                 loDiff=(0.01, 0.01, 0.01),
        #                                 upDiff=(0.01, 0.01, 0.01),
        #                                 mask=None)
        # time4 = time.time()
        # bool_matrix = np.mean(retval, axis=2) == 0
        # bool_zeros = np.zeros(d1 * d2).flatten()
        # bool_indices = np.indices(bool_zeros.shape)[0]  # [::16]
        # bool_zeros[bool_indices] = bool_matrix.flatten()
        # bool_matrix = bool_zeros.reshape((d1, d2))
        #
        # self.curr_ground_coords = np.where(bool_matrix > 0)
        #
        # time5 = time.time()
        # color_image = self.agent.front_rgb_camera.data.copy()
        # time6 = time.time()
        # color_image[bool_matrix > 0] = 255
        # time7 = time.time()
        # cv2.imshow('Color', color_image)
        # cv2.waitKey(1)
        # time8 = time.time()
        # print(f"time12 = {1 / (time2 - time1)} | time23 = {1 / (time3 - time2)} | "
        #       f"time34 = {1 / (time4 - time3)} | time45 = {1 / (time5 - time4)} | time56 = {1 / (time6 - time5)} | "
        #       f"time67 = {1 / (time7 - time6)} | time78 = {1 / (time8 - time7)}"
        #       f"total diff = {1 / (time8 - time1)}")

    def to_world_coords(self, img_ground_coords):
        if self.agent.front_depth_camera.data is not None:
            depth_img = self.agent.front_depth_camera.data
            depths = depth_img[img_ground_coords][:, np.newaxis] * 1000
            result = np.multiply(np.array(img_ground_coords).T, depths)
            raw_p2d = np.hstack((result, depths))
            cords_xyz = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
            # cords_xyz = np.array([
            #     cords_xyz[2, :],
            #     cords_xyz[0, :],
            #     -cords_xyz[1, :],
            #
            # ])
            cords_xyz_1 = np.vstack([cords_xyz, np.ones((1, np.shape(cords_xyz)[1]))])
            result = (self.agent.vehicle.transform.get_matrix() @ cords_xyz_1)[:3, :].T
            # print(np.min(result, axis=0), np.max(result, axis=0))
            return result

    def compute_reference_norm(self, pcd: o3d.geometry.PointCloud):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # build KD tree for fast computation
        [k, idx, _] = pcd_tree.search_knn_vector_3d(self.agent.vehicle.transform.location.to_array(),
                                                    knn=self.knn)  # find points around me
        points_near_me = np.asarray(pcd.points)[idx, :]  # 200 x 3
        u, s, vh = np.linalg.svd(points_near_me, full_matrices=False)  # use svd to find normals of points
        self.reference_norm = vh[2, :]

    @staticmethod
    def normalize_v3(arr):
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        lens[lens <= 0] = 1
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def compute_vectors_near_me(self):
        d1, d2 = self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x
        idx, jdx = np.indices((d1, d2))
        idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
        idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
        jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
        jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()

        # rand_idx = np.random.choice(np.arange(idx.shape[0]), size=d1*d2, replace=False)
        f1 = (idx_front * d2 + jdx)  # [::16]  # [rand_idx]
        f2 = (idx_back * d2 + jdx)  # [::16]  # [rand_idx]
        f3 = (idx * d2 + jdx_front)  # [::16]  # [rand_idx]
        f4 = (idx * d2 + jdx_back)  # [::16]  # [rand_idx]
        return f1, f2, f3, f4

