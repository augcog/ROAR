from ROAR.agent_module.agent import Agent
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np
from typing import Optional, Any
import open3d as o3d
import time, cv2


class GroundPlaneDetector(DepthToPointCloudDetector):
    def __init__(self, agent: Agent, knn: int = 200, res: int = 4, **kwargs):
        super().__init__(agent, **kwargs)
        self.reference_norm: Optional[np.ndarray] = np.array([-0.00000283, -0.00012446, 0.99999999])
        self.knn = knn
        self.res = res
        self.f1, self.f2, self.f3, self.f4 = self.compute_vectors_near_me(res)
        self.threshold = 0.15

    def run_in_series(self) -> Any:
        if self.agent.kwargs.get("point_cloud", None) is not None:
            try:
                points: np.ndarray = self.agent.kwargs.get("point_cloud").copy()
                x = points[self.f3, :] - points[self.f4, :]
                y = points[self.f1, :] - points[self.f2, :]
                normals = self.normalize_v3(np.cross(x, y))
                # OpenCV FloodFill
                d1 = h = self.agent.front_depth_camera.image_size_y
                d2 = w = self.agent.front_depth_camera.image_size_x
                curr_img = normals.reshape((int(d1/self.res), int(d2/self.res), 3)).astype(np.float32)

                min_x, max_x = 0, h // self.res
                min_y, max_y = w * 3 // 4 // self.res, w

                # Y_norm_array: np.ndarray = curr_img[min_x:max_x, min_y:max_y, 1]
                # x, y = np.unravel_index(np.argmax(Y_norm_array), np.shape(Y_norm_array))
                # seed_w, seed_h = y + min_y, x + min_x
                # print(seed_w, seed_h, np.shape(curr_img))

                seed_point = (int(d1/self.res) - 10, int(int(d2/self.res) / 2))
                _, retval, _, _ = cv2.floodFill(image=curr_img,
                                                seedPoint=seed_point,
                                                newVal=(0, 0, 0),
                                                loDiff=(self.threshold,self.threshold,self.threshold),
                                                upDiff=(self.threshold,self.threshold,self.threshold),
                                                mask=None,
                                                flags=8)
                bool_matrix = np.mean(retval, axis=2) == 0
                bool_zeros = np.zeros(d1 * d2).flatten()
                bool_indices = np.indices(bool_zeros.shape)[0][::self.res**2]
                bool_zeros[bool_indices] = bool_matrix.flatten()
                bool_matrix = bool_zeros.reshape((d1, d2))

                color_image = self.agent.front_rgb_camera.data.copy()
                color_image[bool_matrix > 0] = 255
                cv2.imshow('Color', color_image)
                cv2.waitKey(1)
            except Exception as e:
                self.logger.error(e)

    @staticmethod
    def construct_pointcloud(points) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        return pcd

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


    def compute_vectors_near_me(self, res):
        d1, d2 = self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x
        idx, jdx = np.indices((d1, d2))
        idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
        idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
        jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
        jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()

        # rand_idx = np.random.choice(np.arange(idx.shape[0]), size=d1*d2, replace=False)
        f1 = (idx_front * d2 + jdx)[::res**2]  # [rand_idx]
        f2 = (idx_back * d2 + jdx)[::res**2]  # [rand_idx]
        f3 = (idx * d2 + jdx_front)[::res**2]  # [rand_idx]
        f4 = (idx * d2 + jdx_back)[::res**2]  # [rand_idx]
        return f1, f2, f3, f4