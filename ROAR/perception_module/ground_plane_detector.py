from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional, Any, Tuple
import open3d as o3d
import time, cv2
from ROAR.utilities_module.utilities import img_to_world
from scipy import stats


class GroundPlaneDetector(Detector):
    def __init__(self, agent: Agent, knn: int = 200, roi: Optional[Tuple[Tuple[int, int]]] = None, **kwargs):
        super().__init__(agent, **kwargs)
        self.knn = knn
        self.f1, self.f2, self.f3, self.f4 = self.compute_vectors_near_me()
        self.roi = roi if roi is not None else ((0, self.agent.front_depth_camera.image_size_x),
                                                (self.agent.front_depth_camera.image_size_y * 3 // 4,
                                                 self.agent.front_depth_camera.image_size_y))
        self.min_x, self.max_x = self.roi[0][0], self.roi[0][1]
        self.min_y, self.max_y = self.roi[1][0], self.roi[1][1]
        self.threshold = 0.15

    def run_in_threaded(self, **kwargs):
        while True:
            self.agent.kwargs["ground_coords"] = self.run_in_series()

    def run_in_series(self) -> Any:
        if self.agent.kwargs.get("point_cloud", None) is not None:
            try:
                points: np.ndarray = self.agent.kwargs.get("point_cloud")
                # from points find normal vectors
                h = self.agent.front_depth_camera.image_size_y
                w = self.agent.front_depth_camera.image_size_x
                # start of efficiency bottle neck
                x = points[self.f1, :] - points[self.f2, :]
                y = points[self.f3, :] - points[self.f4, :]
                xyz_norm = self.normalize_v3(np.cross(x, y))
                # end of efficiency bottle neck

                # reshape and make-positive the normal vector since directions don't matter for ground plane detection
                xyz_norm = np.abs(xyz_norm)
                xyz_norm = xyz_norm.reshape((h, w, 3)).astype(np.float32)

                # we only need to consider the a single axis norm
                Y_norm_array: np.ndarray = xyz_norm[self.min_x:self.max_x, self.min_y:self.max_y, 0]
                x, y = np.unravel_index(np.argmax(Y_norm_array), np.shape(Y_norm_array))
                seed_h, seed_w = y + self.min_y, x + self.min_x

                # floodfill
                mask = np.zeros((xyz_norm.shape[0] + 2, xyz_norm.shape[1] + 2), np.uint8)
                fillvalue = 255
                cv2.floodFill(image=xyz_norm, mask=mask, seedPoint=(seed_w, seed_h), newVal=fillvalue,
                              loDiff=(self.threshold, self.threshold, self.threshold),
                              upDiff=(self.threshold, self.threshold, self.threshold),
                              flags=8 | (fillvalue << 8) | cv2.FLOODFILL_MASK_ONLY)
                mask = mask[1:-1, 1:-1]
                mask[self.agent.front_depth_camera.data > 0.5] = 0
                cv2.imshow("mask original", mask)

                ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # You need to choose 4 or 8 for connectivity type
                connectivity = 4
                # Perform the operation
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity,
                                                                                        cv2.CV_32S)
                # find the label with the biggest area
                areas = stats[:, 4]
                nr = np.arange(num_labels)
                ground_area, ground_label = sorted(zip(areas, nr), reverse=True)[1]
                if ground_area < 10000:
                    return
                mask = np.zeros(labels.shape)
                mask[labels == ground_label] = 1
                cv2.imshow("mask connected component", mask)

                kernel = np.ones((5, 5), np.uint8)
                closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                cv2.imshow('mask closing', closing)

                # Y_map = cv2.normalize(Y_norm_array, None, 1, 0, cv2.NORM_MINMAX)
                # cv2.imshow("Normal", Y_map)
                cv2.imshow("depth", self.agent.front_depth_camera.data)
                # cv2.imshow("Mask", mask)
                # color_image = self.agent.front_rgb_camera.data.copy()
                # color_image = cv2.rectangle(color_image,
                #                             pt1=(self.min_x, self.min_y),
                #                             pt2=(self.max_x, self.max_y),
                #                             color=(255, 0, 0),
                #                             thickness=10)
                # color_image = cv2.circle(color_image, (seed_w, seed_h), 6, (255, 0, 0), 2)
                # cv2.imshow("color", color_image)
                cv2.waitKey(1)
            except Exception as e:
                self.logger.error(f"Failed to find ground plane: seed point = {e}")

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
