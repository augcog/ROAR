from typing import Any
from ROAR.perception_module.detector import Detector
from numpy import log, exp
from scipy.optimize.minpack import leastsq
import numpy as np
import cv2
import logging
from scipy.signal import convolve2d
from typing import Optional


class SemanticSegmentationDetector(Detector):
    SKY = [255, 0, 0]
    GROUND = [255, 255, 255]
    OBSTACLE = [0,0,0]

    def __init__(self, sky_level=0.9, t=0.05, del_ang=0.2, fit_type='exp', **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("Ground Plane Detector")
        self.sky_level = sky_level
        self.thresh = t
        self.fit_type = fit_type
        self.del_ang = del_ang
        self.roll_ang = 0
        self.rot_axis = [0, 0, 1]

        self.orig_preds = None
        self.preds = None

        self.curr_segmentation: Optional[np.ndarray] = None
        self.logger.info("Ground Plane Detector Initiated")

    @staticmethod
    def convert_to_log(x):
        return np.clip(1 + np.log(x + 1e-10) / 5.70378, 0.005, 1.0)

    def run_in_series(self) -> Any:
        logged_depth = self.convert_to_log(self.agent.front_depth_camera.data.copy())
        if self.orig_preds is None or self.preds is None:
            self.orig_preds = self.gpd_mesh(logged_depth)
            self.preds = np.copy(self.orig_preds)
            self.logger.debug("Ground Plane Preds Computed")
        else:
            self.curr_segmentation = self.output_gpd(logged_depth)

    def gpd_mesh(self, depth_image):
        xs = []
        data = []
        max_depth = 0
        for i in range(depth_image.shape[0] - 1, -1, -1):
            j = np.argmax(depth_image[i, :])
            d = depth_image[i][j]
            if d > 0.3:
                break
            if d > max_depth and d > 0.01:
                max_depth = d
                xs.append(i)
                data.append(d)

        xs = np.array(xs[::-1], dtype=np.float64)
        data = np.array(data[::-1], dtype=np.float64)

        if self.fit_type == 'lsq':
            a, b, c, d = _Leastsq_Exp.fit(xs / xs.max(), data)
            pred_func = _Leastsq_Exp.construct_f(a, b, c, d)
            rows = np.meshgrid(
                np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
            )[1]
            preds = pred_func(rows / rows.max())
            preds[preds > 1] = 0
            return preds
        else:
            a, b, c, p, q = _Exponential_Model.fit(xs, data)
            pred_func = _Exponential_Model.construct_f(a, b, c, p, q)
            rows = np.meshgrid(
                np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
            )[1]
            preds = pred_func(rows)
            preds[preds > 1] = 0
            return preds

    def reg_img_to_world(self, depth_img, sky_level=0.9, depth_scaling_factor=1000) -> np.ndarray:
        # (Intrinsic) K Matrix

        intrinsics_matrix = self.agent.front_depth_camera.intrinsics_matrix

        # get a 2 x N array for their indices
        bool_mat = depth_img > (depth_img.min() - 1)
        ground_loc = np.where(bool_mat)
        depth_val = depth_img[bool_mat] * depth_scaling_factor
        ground_loc = ground_loc * depth_val

        # compute raw_points
        raw_points = np.vstack([ground_loc, depth_val])

        return np.linalg.inv(intrinsics_matrix) @ raw_points

    def img_to_world(self, depth_img, sky_level=0.2, depth_scaling_factor=1000) -> np.ndarray:
        # (Intrinsic) K Matrix
        intrinsics_matrix = self.agent.front_depth_camera.intrinsics_matrix

        # get a 2 x N array for their indices
        bool_mat3 = (0.1 < depth_img) * (depth_img < sky_level)
        # bool_mat2 = 0.1 < depth_img
        # bool_mat3 = bool_mat * bool_mat2
        ground_loc = np.where(bool_mat3)
        depth_val = depth_img[bool_mat3] * depth_scaling_factor
        ground_loc = ground_loc * depth_val

        # compute raw_points
        raw_points = np.vstack([ground_loc, depth_val])

        # convert to cords_y_minus_z_x
        return np.linalg.inv(intrinsics_matrix) @ raw_points

    def get_roll_stats(self, depth_image):
        xyz = self.img_to_world(depth_image).T
        xyz_samp = xyz[np.random.choice(xyz.shape[0], 400, replace=False), :]
        u, s, vt = np.linalg.svd(xyz_samp - xyz_samp.mean())

        reg = vt[2]
        no_y = np.array(reg, copy=True)
        no_y[1] = 0
        nvt, nx = np.linalg.norm(reg), np.linalg.norm(no_y)
        cos_ang = np.dot(reg, no_y) / (nvt * nx)

        unitcross = lambda a, b: np.cross(a, b) / np.linalg.norm(np.cross(a, b))
        rot_axis = unitcross(vt[2], no_y)

        return np.arccos(cos_ang), rot_axis  # radians

    def roll_frame(self, depth_image, ang, rot_axis, no_axis=False):
        if no_axis:
            return depth_image

        xyz = self.reg_img_to_world(depth_image).T
        xyz_mean = xyz.mean(axis=0)
        xyz = xyz - xyz_mean

        cos_ang = np.cos(ang)
        x, y, z = rot_axis

        c = cos_ang
        s = np.sqrt(1 - c * c)
        C = 1 - c
        rmat = np.array([[x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                         [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                         [z * x * C - y * s, z * y * C + x * s, z * z * C + c]])
        rot_xyz = rmat @ (xyz.T)
        rot_xyz = rot_xyz.T + xyz_mean
        return rot_xyz[:, 2].reshape(depth_image.shape) / 1000

    def output_gpd(self, d_frame):
        # first im going to find out where is the sky
        sky = np.where(d_frame > self.sky_level)

        # then im going to find out where is the ground
        ground = np.where(np.abs(d_frame - self.preds) < self.thresh)

        result = np.zeros(shape=(d_frame.shape[0], d_frame.shape[1], 3))
        result[ground] = self.GROUND
        result[sky] = self.SKY
        # result = result.astype('uint8')
        # # try:
        # #     new_roll_ang, self.rot_axis = self.get_roll_stats(d_frame)  # this method is pretty slow
        # #     if np.abs(self.roll_ang - new_roll_ang) > self.del_ang:
        # #         print(f"Recalibrating {self.agent.time_counter}")
        # #         self.roll_ang = new_roll_ang
        # #         self.preds = self.roll_frame(self.orig_preds, self.roll_ang, -1 * self.rot_axis)
        # # except Exception as e:
        # #     self.logger.error(f"Failed to compute output: {e}")
        #
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # # img = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1]
        # ret, thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # print(np.shape(img))
        # # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image=thresh, connectivity=8,ltype=cv2.CV_16U) # img= (600 x 800)
        # _, labels = cv2.connectedComponents(image=thresh)
        # indicies = np.where(labels == 2)
        # print(np.shape(indicies))
        # #
        # cv2.imshow("segmented", result)
        # cv2.waitKey(1)
        return result


class _Exponential_Model:

    @staticmethod
    def Sk(S_k_1, y_k, y_k_1, x_k, x_k_1):
        return S_k_1 + 0.5 * (y_k + y_k_1) * (x_k - x_k_1)

    @staticmethod
    def SSk(SS_k_1, S_k, S_k_1, x_k, x_k_1):
        return SS_k_1 + 0.5 * (S_k + S_k_1) * (x_k - x_k_1)

    @staticmethod
    def S(x, y):
        ret_S = [0]
        for k in range(1, len(x)):
            S_k = _Exponential_Model.Sk(ret_S[k - 1], y[k], y[k - 1], x[k], x[k - 1])
            ret_S.append(S_k)
        return ret_S

    @staticmethod
    def SS(s, x):
        ret_SS = [0]
        for k in range(1, len(x)):
            SS_k = _Exponential_Model.SSk(ret_SS[k - 1], s[k], s[k - 1], x[k], x[k - 1])
            ret_SS.append(SS_k)
        return ret_SS

    @staticmethod
    def F1(SS_k, y_k):
        return SS_k / y_k

    @staticmethod
    def F2(S_k, y_k):
        return S_k / y_k

    @staticmethod
    def F3(x_k, y_k):
        return (x_k ** 2) / y_k

    @staticmethod
    def F4(x_k, y_k):
        return x_k / y_k

    @staticmethod
    def F5(y_k):
        return 1 / y_k

    @staticmethod
    def construct_f(a, b, c, p, q):
        def f(x):
            # print(type(a), type(b), type(c), type(p), type(q))
            return a + b * np.exp(p * x) + c * np.exp(q * x)

        return f

    @staticmethod
    def fit(x, y):
        Sxy = _Exponential_Model.S(x, y)
        SSxy = _Exponential_Model.SS(Sxy, x)
        F1xy = _Exponential_Model.F1(SSxy, y)
        F2xy = _Exponential_Model.F2(Sxy, y)
        F3xy = _Exponential_Model.F3(x, y)
        F4xy = _Exponential_Model.F4(x, y)
        F5xy = _Exponential_Model.F5(y)
        F = np.array([F1xy, F2xy, F3xy, F4xy, F5xy])
        f = np.array([np.sum(F1xy), np.sum(F2xy), np.sum(F3xy),
                      np.sum(F4xy), np.sum(F5xy)])
        F = F @ F.T
        A, B, C, D, E = np.linalg.inv(F) @ f
        pre_sqrt = np.clip(B ** 2 + 4 * A, 0, np.inf)  # edits 1

        p = 0.5 * (B + np.sqrt(pre_sqrt))
        q = 0.5 * (B - np.sqrt(pre_sqrt))
        G1 = 1 / y
        G2 = np.exp(p * x) / y
        G3 = np.exp(q * x) / y
        G = np.array([G1, G2, G3])
        G = G @ G.T
        g = np.array([np.sum(G1), np.sum(G2), np.sum(G3)])
        a, b, c = np.linalg.pinv(G) @ g  # edits 2
        return a, b, c, p, q


class _Leastsq_Exp:
    @staticmethod
    def f(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)

    # def fit(x, y):
    #     a, b, c, d = curve_fit(f, x, y, p0=(1, 1, 1, 1))
    #     return a, b, c, d

    ## regression function
    @staticmethod
    def _exp(a, b, c, d):
        """
        Exponential function y = a * exp(b * x) + c * exp(d * x)
        """
        return lambda x: a * exp(b * x) + c * exp(d * x)

    @staticmethod
    def fit(x, y):
        fun = _Leastsq_Exp._exp
        df = {'x': x, 'y': y}
        resid = lambda p, x, y: y - fun(*p)(x)
        ls = leastsq(resid, np.array([1.0, 1.0, 1.0, 1.0]), args=(df['x'], df['y']))
        a, b, c, d = ls[0]
        return a, b, c, d

    @staticmethod
    def construct_f(a, b, c, d):
        def f(x):
            # print(type(a), type(b), type(c), type(p), type(q))
            return a * np.exp(b * x) + c * np.exp(d * x)

        return f

    ## interpolation
    @staticmethod
    def interpolate(x, df, fun=None):
        """
        Interpolate Y from X based on df, a dataframe with columns 'x' and 'y'.
        """
        if fun is None:
            fun = _Leastsq_Exp._exp
        resid = lambda p, x, y: y - fun(*p)(x)
        ls = leastsq(resid, np.array([1.0, 1.0, 1.0, 1.0]), args=(df['x'], df['y']))
        a, b, c, d = ls[0]
        y = fun(a, b, c, d)(x)
        return y, a, b, c, d
