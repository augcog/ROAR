from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional, Any
import open3d as o3d
import time, cv2
from ROAR.utilities_module.utilities import img_to_world
import open3d as o3d
import pyransac3d as pyrsc
from scipy import stats

class GroundPlaneDetector(Detector):
    def __init__(self, agent: Agent, knn: int = 200, horizon_row=400, **kwargs):
        super().__init__(agent, **kwargs)
        self.horizon_row = horizon_row
        self.knn = knn
        self.f1, self.f2, self.f3, self.f4 = self.compute_vectors_near_me()

    def run_in_threaded(self, **kwargs):
        while True:
            self.agent.kwargs["ground_coords"] = self.run_in_series()

    def run_in_series(self) -> Any:
        if self.agent.kwargs.get("point_cloud", None) is not None:
            points: np.ndarray = self.agent.kwargs.get("point_cloud")

            h = self.agent.front_depth_camera.image_size_y
            w = self.agent.front_depth_camera.image_size_x

            # cv2.imshow("depth", self.agent.front_depth_camera.data / np.amax(self.agent.front_depth_camera.data))
            x = points[self.f1, :] - points[self.f2, :]
            y = points[self.f3, :] - points[self.f4, :]
            xyz_norm = self.normalize_v3(np.cross(x, y))
            xyz_norm = np.abs(xyz_norm)
            xyz_norm = xyz_norm.reshape((h, w, 3)).astype(np.float32)
            # print(stats.describe(xyz_norm[:,:, 2].flatten()))
            Y_array = xyz_norm[-10, :, 1]
            seed_w = np.argsort(Y_array)[len(Y_array) // 2]
            seed_h = xyz_norm.shape[0] - 25
            # seed_point = (xyz_norm.shape[1] // 2, xyz_norm.shape[0] - 25)  # (d1 - 2, int(d2 / 2))
            # print(seed_point)
            threshold = 0.01
            mask = np.zeros((xyz_norm.shape[0] + 2, xyz_norm.shape[1] + 2), np.uint8)
            fillvalue = 255
            cv2.floodFill(image=xyz_norm, mask=mask, seedPoint=(seed_w, seed_h), newVal=fillvalue,
                          loDiff=(threshold, threshold, threshold), upDiff=(threshold, threshold, threshold),
                          flags=8 | (fillvalue << 8) | cv2.FLOODFILL_MASK_ONLY)
            normalized_depth = self.agent.front_depth_camera.data / np.max(self.agent.front_depth_camera.data)
            idx = (normalized_depth >= 0.01)

            mask = mask[:-2, :-2]
            # print(np.shape(idx), np.shape(mask))
            mask[idx] = 0
            # mask[xyz_norm > xyz_norm[:, :, 2]:wq
            # .flatten().mean()] = 0
            # mask[xyz_norm[xyz_norm > xyz_norm[:,:,2].flatten().mean()]] = 0
            cv2.imshow("Ground Plane Mask", mask)
            cv2.waitKey(1)

            # # print(np.amin(points, 0), np.amax(points, 0))
            # horizon_row = 200
            # d1, d2 = int(480 / 1), int(640 / 1)
            # idx, jdx = np.indices((d1, d2))
            # idx, jdx = idx[horizon_row:, :], jdx[horizon_row:, :]
            # idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
            # idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
            # jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
            # jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
            # idx = idx.flatten()
            # jdx = jdx.flatten()
            #
            # f1 = (idx_front * d2 + jdx)
            # f2 = (idx_back * d2 + jdx)
            # f3 = (idx * d2 + jdx_front)
            # f4 = (idx * d2 + jdx_back)
            #
            # x = points[f1, :] - points[f2, :]
            # y = points[f3, :] - points[f4, :]
            # xyz_norm = self.normalize_v3(np.cross(x, y))
            #
            # # OpenCV FloodFill
            # curr_img = xyz_norm.reshape((d1 - horizon_row, d2, 3)).astype(np.float32)
            # mask = np.zeros((curr_img.shape[0] + 2, curr_img.shape[1] + 2), np.uint8)
            # seed_point = (curr_img.shape[1] // 2, curr_img.shape[0] - 25)  # (d1 - 2, int(d2 / 2))
            # # print(color_image.shape, curr_img.shape, mask.shape, seed_point)
            # threshold = 0.01
            # _, retval, _, _ = cv2.floodFill(image=curr_img,
            #                                 seedPoint=seed_point,
            #                                 newVal=(0, 0, 0),
            #                                 loDiff=(threshold, threshold, threshold),
            #                                 upDiff=(threshold, threshold, threshold),
            #                                 mask=mask, flags=8|(255<<8)|cv2.FLOODFILL_MASK_ONLY)
            #
            # norm_umatrix = np.zeros((d1, d2))
            # norm_umatrix[horizon_row:, :] = mask[1:-1, 1:-1] * 255
            # color_image = self.agent.front_rgb_camera.data
            # color_image[norm_umatrix > 0] = 255  # [out.get() != max_label] = 255
            #
            # cv2.imshow('Color', color_image)
            # cv2.waitKey(1)

            # d1, d2 = self.agent.front_depth_camera.image_size_y, self.agent.front_depth_camera.image_size_x
            # x = points[self.f1, :] - points[self.f2, :]
            # y = points[self.f3, :] - points[self.f4, :]
            # xyz_norm = self.normalize_v3(np.cross(x, y))
            # curr_img = xyz_norm.reshape((d1 - self.horizon_row, d2, 3)).astype(np.float32)
            # mask = np.zeros((curr_img.shape[0] + 2, curr_img.shape[1] + 2), np.uint8)
            # seed_point = (curr_img.shape[1] // 2, curr_img.shape[0] - 25)  # (d1 - 2, int(d2 / 2))
            #
            # _, retval, _, _ = cv2.floodFill(image=curr_img,
            #                                 seedPoint=seed_point,
            #                                 newVal=(0, 0, 0),
            #                                 loDiff=(0.25, 0.25, 0.25),
            #                                 upDiff=(0.25, 0.25, 0.25),
            #                                 mask=mask)
            #
            # norm_umatrix = np.zeros((d1, d2))
            # norm_umatrix[self.horizon_row:, :] = mask[1:-1, 1:-1] * 255
            # color_image = self.agent.front_rgb_camera.data
            # color_image[norm_umatrix > 0] = 255
            #
            # cv2.imshow('Color', color_image)
            # cv2.waitKey(1)

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


def computer_normal(normal_window):
    normal = np.zeros((3,))
    # Extract only nonzero normals from the normal_window
    wh, ww, _ = normal_window.shape
    point_list = []
    for h_index in range(wh):
        for w_index in range(ww):
            if normal_window[h_index, w_index, 2] != 0:
                point_list.append(normal_window[h_index, w_index, :])

    if len(point_list) > 3:
        vector_list = []
        for index1 in range(len(point_list) - 1):
            for index2 in range(index1 + 1, len(point_list)):
                vector_list.append(point_list[index1] - point_list[index2])
        normal_array = np.vstack(vector_list)
        U, S, Vh = np.linalg.svd(normal_array)
        normal = Vh[-1, :]

        # Normal may point to opposite directions
        # For ground-plane detection, we force positive Y direction
        if normal[1] < 0:
            normal = -normal

    return normal
