from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional, Any, Tuple
import open3d as o3d
import time, cv2


class ObstacleDetector(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, knn: int = 200, roi: Optional[Tuple[Tuple[int, int]]] = None, **kwargs):
        super().__init__(agent, **kwargs)
        self.knn = knn
        self.f1, self.f2, self.f3, self.f4 = self.compute_vectors_near_me()
        self.roi = roi if roi is not None else ((0, self.agent.front_depth_camera.image_size_x),
                                                (self.agent.front_depth_camera.image_size_y * 3 // 4,
                                                 self.agent.front_depth_camera.image_size_y))
        self.min_x, self.max_x = self.roi[0][0], self.roi[0][1]
        self.min_y, self.max_y = self.roi[1][0], self.roi[1][1]
        self.threshold = 0.01
        self.curr_mask = None

    def run_in_threaded(self, **kwargs):
        while True:
            self.run_in_series()

    def run_in_series(self) -> Any:
        if self.agent.kwargs.get("point_cloud", None) is not None:
            try:
                points: np.ndarray = self.agent.kwargs.get("point_cloud").copy()
                depth_data: np.ndarray = self.agent.front_depth_camera.data.copy()
                # print("received pointcloud", np.amin(points, axis=0), np.amax(points, axis=0), self.agent.vehicle.transform.location)
                # from points find normal vectors
                h = self.agent.front_depth_camera.image_size_y
                w = self.agent.front_depth_camera.image_size_x

                # start of efficiency bottle neck TODO: @christian
                x = points[self.f1, :] - points[self.f2, :]
                y = points[self.f3, :] - points[self.f4, :]
                xyz_norm = self.normalize_v3(np.cross(x, y))
                # end of efficiency bottle neck

                # reshape and make-positive the normal vector since directions don't matter for ground plane detection
                xyz_norm = np.abs(xyz_norm)
                xyz_norm = xyz_norm.reshape((h, w, 3)).astype(np.float32)

                # we only need to consider the a single axis norm
                Y_norm_array: np.ndarray = xyz_norm[self.min_x:self.max_x, self.min_y:self.max_y, 1]
                x, y = np.unravel_index(np.argmax(Y_norm_array), np.shape(Y_norm_array))
                seed_h, seed_w = y + self.min_y, x + self.min_x

                # floodfill
                ground_mask = np.zeros((xyz_norm.shape[0] + 2, xyz_norm.shape[1] + 2), np.uint8)
                fillvalue = 255
                cv2.floodFill(image=xyz_norm, mask=ground_mask, seedPoint=(seed_w, seed_h), newVal=fillvalue,
                              loDiff=(self.threshold, self.threshold, self.threshold),
                              upDiff=(self.threshold, self.threshold, self.threshold),
                              flags=8 | (fillvalue << 8) | cv2.FLOODFILL_MASK_ONLY)
                ground_mask = ground_mask[1:-1, 1:-1]
                sky_mask = depth_data > 0.1
                ground_mask[sky_mask] = 0

                ret, thresh = cv2.threshold(ground_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # You need to choose 4 or 8 for connectivity type
                connectivity = 8
                # Perform the operation
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=thresh,
                                                                                        connectivity=connectivity,
                                                                                        ltype=cv2.CV_32S)
                # find the label with the biggest area
                nr = np.arange(num_labels)
                ground_area, ground_label = sorted(zip(stats[:, 4], nr), reverse=True)[1]
                if ground_area < 10000:
                    return
                ground_mask = np.zeros(labels.shape)
                ground_mask[labels == ground_label] = 1
                ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE,  np.ones((5, 5), np.uint8))

                obstacle_mask = np.ones(shape=depth_data.shape)
                obstacle_mask[sky_mask] = 0
                obstacle_mask[ground_mask == 1] = 0
                obstacle_mask[:depth_data.shape[1] // 8 * 3, :] = 0
                self.curr_mask = obstacle_mask
                # cv2.imshow("obstacle mask", obstacle_mask)

                xyz = np.reshape(a=points, newshape=(h, w, 3))
                ground_coords = xyz[ground_mask == 1]
                self.agent.kwargs["ground_coords"] = ground_coords

                obstacle_coords = xyz[obstacle_mask == 1]
                vehicle_location = self.agent.vehicle.transform.location.to_array()
                dists = np.linalg.norm(obstacle_coords - vehicle_location, axis=1)
                obstacle_coords = obstacle_coords[dists < 100]  # consider doing this filter early on

                self.agent.kwargs["obstacle_coords"] = obstacle_coords

            except Exception as e:
                self.logger.error(f"Failed to find ground plane: {e}")

    @staticmethod
    def normalize_v3(arr):
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        lens[lens <= 0] = 1
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def compute_vectors_near_me(self):
        """Computes vectors near Agent from Front Depth Camera."""
        d1, d2 = self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x
        idx, jdx = np.indices((d1, d2))
        # idx, jdx = idx[self.horizon_row:, :], jdx[self.horizon_row:, :]
        idx, jdx = idx[:, :], jdx[:, :]
        idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
        idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
        jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
        jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()

        f1 = (idx_front * d2 + jdx)
        f2 = (idx_back * d2 + jdx)
        f3 = (idx * d2 + jdx_front)
        f4 = (idx * d2 + jdx_back)
        return f1, f2, f3, f4
