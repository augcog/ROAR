import json
import time
import math
import os
from typing import Any

from cv2 import aruco
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.perception_module.detector import Detector


class ARMarkerLocalizationDetector(Detector):
    """
    allow for the car to locate its global positioning
    """

    def __init__(self, agent: Agent, **kwargs):
        """

        Args:
            agent: ROAR Agent instance
        """

        # Parse the json to get the map
        super().__init__(agent, **kwargs)
        self.json_in_path: Path = Path(agent.agent_settings.json_qr_code_file_path)
        content = self.json_in_path.open('r')
        self.agent = agent
        self.json_in = json.loads(content.read())
        self.map = self.json_in["Segments"]
        self.ar_tags = self.json_in["AR tags"]
        self.ar_configs = {}

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # Store the latest useful graysacle image and depth information
        self.prev_gray = None
        self.prev_depth = None

        # objects used for keyframe matching
        self.fast = cv2.FastFeatureDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.ARglobal2local = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.prev_config = None
        self.config_AR()

        self.running = True

    def config_AR(self):
        """
        compute the configuration of the local coordinates of ar tags with respect to world frame
        :return: a dictionary of (ar_id, config)
        """
        for i in self.ar_tags:
            t = i["Location"][:3]
            ar_rpy = i["Location"][3:]

            # compute the rotation matrix, assuming extrinsic rotations
            roll, pitch, yaw = np.deg2rad(ar_rpy[0]), np.deg2rad(ar_rpy[1]), np.deg2rad(ar_rpy[2])
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
            Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
            R = np.dot(Rz, np.dot(Ry, Rx))

            # change to ar tag's local coordinates
            R = np.dot(R, self.ARglobal2local)

            t = np.mat(t).T
            config = np.hstack((R, t))
            config = np.vstack((config, np.mat([0, 0, 0, 1])))
            self.ar_configs[i["Id"]] = config

    def get_global_position(self):
        """
        Compute the location of the car in the global frame
        :return: a array of (x, y, z, roll, pitch, yaw)
        """
        if self.prev_config is None:
            return None

        roll, pitch, yaw = cv2.decomposeProjectionMatrix(self.prev_config[:3])[-1].reshape((3,))
        x, y, z = self.prev_config[:3, 3]
        return np.array([x, y, z, roll, pitch, yaw])

    def run_in_series(self, **kwargs) -> Any:
        pass

    def run_in_threaded(self):
        """
            Compute the configuration of the car
        """
        while self.running:
            img_arr = None if self.agent.front_rgb_camera.data is None else self.agent.front_rgb_camera.data.copy()
            depth_arr = None if self.agent.front_depth_camera.data is None else self.agent.front_depth_camera.data.copy()

            if type(img_arr) == np.ndarray:
                if not img_arr.size:
                    continue
            else:
                continue
            cv2.imshow("img from ar marker localization", img_arr)
            cv2.waitKey(1)
            # Copy image then convert it to grayscale
            img_arr = img_arr.copy()
            if len(img_arr) == 3:
                gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_arr

            # Get the position list from the gray scale image, if there are ar tags
            positions_list = self.get_position_list(gray)

            if positions_list is not None:
                # Average out the ar tag positions based on a weighted average
                self.prev_config = np.array(self.avg_RT(positions_list))
                self.prev_gray, self.prev_depth = gray, depth_arr
                continue

            elif self.prev_gray is not None and img_arr is not None and depth_arr is not None:  # KEYFRAME MATCHING
                if self.prev_config is None:
                    continue

                R, T = self.keyframe_matching(gray, depth_arr)
                if T is None or R is None:
                    # self.cur_gray, self.cur_depth = None, None
                    return None, False

                T = T / 1000  # Normalize millimeters to meters
                g = np.hstack((R, T))
                g = np.vstack((g, [0, 0, 0, 1]))

                self.prev_config = np.dot(self.prev_config, g)
                self.prev_gray, self.prev_depth = gray, depth_arr

    def get_position_list(self, gray=None):  # Get list of all global positions based on each ar tag in the frame
        """
        Compute a list of positions based on ar tags seen in the frame
        :param gray: the input image
        :return a list of [config matrix, dist]
        """
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        # Detect the ids
        corners, ids, rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=parameters)
        if np.all(ids != None):
            valid_ids = True
            # Get rvec and tvec of each id relative to detected ar tags
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.2032,
                                                              self.agent.front_rgb_camera.intrinsics_matrix,
                                                              self.agent.front_rgb_camera.distortion_coefficient)

            config_list, dists = [], []
            for i, val in enumerate(ids):
                curr_time = time.ctime(time.time())
                curr_id = val[0]

                # Get the distances and angles
                rvec = rvecs[i]
                tvec = tvecs[i]

                # Compute the car's position relative to ar tag viewed in ar tag's local coordinates
                rmat = cv2.Rodrigues(rvec)[0]
                P = np.hstack((rmat, tvec.T))
                P = np.vstack((P, [0, 0, 0, 1]))
                ARlocal2Car = np.linalg.inv(P)

                # Compute the car's position viewed in the world frame
                World2ARlocal = self.ar_configs[curr_id]
                World2Car = np.dot(World2ARlocal, ARlocal2Car)

                config_list.append(World2Car)
                dist = np.linalg.norm(tvec)
                dists.append(dist)

            # Format positions list correctly
            positions_list = list(zip(config_list, dists))
            return positions_list
        return None

    def avg_RT(self, positions):
        """
        Compute the weighted average of config metrices, using linear interpolation on T
        and spherical interpolation on R
        :param positions: a list of [config matrix, dist]
        :return: the weighted average
        """
        if not positions:
            return positions
        if len(positions) == 1:
            return positions[0][0]

        # Initialize weights = 1/distance for the interpolation
        lastWeight = 1 / positions[0][1]
        lastR = positions[0][0][0:3, 0:3]
        lastT = positions[0][0][0:3, 3]

        # Iteratively recompute weighted average
        for i in range(1, len(positions)):
            newWeight = 1 / positions[i][1]
            newT = positions[i][0][0:3, 3]
            newR = positions[i][0][0:3, 0:3]

            # Linearly interpolate on T
            lastT = (lastWeight * lastT + newWeight * newT) / (lastWeight + newWeight)

            # Spherical interpolation on R
            inp = R.from_matrix(np.array([lastR, newR]))
            slerp = Slerp([0, 1], inp)
            lastR = slerp([(newWeight) / (lastWeight + newWeight)]).as_matrix()[0]

            # Change weight
            lastWeight += newWeight
        tmp = np.hstack((lastR, lastT))
        ret = np.vstack((tmp, [0, 0, 0, 1]))
        return ret

    def keyframe_matching(self, cur_gray, cur_depth):
        """
        Using keyframe matching to find the transformation between self.prev_gray and self.cur_gray.
        :param cur_gray: the input grayscale image
        :param cur_depth: the input depth information
        :return: the transformation info, rotation and translation
        """
        # find the feature descriptors and keypoints
        prev_kp, prev_des, cur_kp, cur_des = self.feature_extraction(cur_gray)

        # feature matching using brute-force and hamming distance
        # prev_gray is the train image and cur_gray is the query image
        matches = self.bf.match(cur_des, prev_des)

        # compute local 3D points
        cur_p3d, prev_p3d = self.compute3D(matches, cur_kp, prev_kp, cur_depth)

        # estimate the rigid transformation
        R, t = self.estimate_rigid_transformation(cur_p3d, prev_p3d)

        return R, t

    def feature_extraction(self, cur_gray):
        """
        Extract the features of self.prev_gray and cur_gray using FAST detector and BRIEF
        :param cur_gray: the input grayscale image
        :return: key points and descriptors of self.prev_gray and cur_gray
        """

        # find the keypoints with FAST and descriptors of self.prev_gray
        kp1 = self.fast.detect(self.prev_gray, None)
        kp1, des1 = self.brief.compute(self.prev_gray, kp1[:200])

        # find the keypoints with FAST and descriptors of cur_gray
        kp2 = self.fast.detect(cur_gray, None)
        kp2, des2 = self.brief.compute(cur_gray, kp2[:200])

        return kp1, des1, kp2, des2

    def compute3D(self, matches, cur_kp, prev_kp, cur_depth):
        """
        compute the local 3d points
        :param matches: the feature matched using BFmatching
        :param prev_kp: the keypoints in self.prev_gray
        :param cur_kp: the keypoints in the input grayscale image
        :param cur_depth: the depth information of the input image
        :return: ndarray of points in 3D
        """
        # keypoints of current image, previous image in homogeneous coordinates  3 x n
        cur_2d = np.asarray([[cur_kp[match.queryIdx].pt[0], cur_kp[match.queryIdx].pt[1], 1] for match in matches]).T
        prev_2d = np.asarray([[prev_kp[match.trainIdx].pt[0], prev_kp[match.trainIdx].pt[1], 1] for match in matches]).T

        # get the depth info of each point, n x n
        current_depth = np.diag(cur_depth[cur_2d[1].astype(int), cur_2d[0].astype(int)])
        previous_depth = np.diag(self.prev_depth[prev_2d[1].astype(int), prev_2d[0].astype(int)])

        return (np.linalg.pinv(self.agent.front_depth_camera.intrinsics_matrix) @ cur_2d @ current_depth).T, (
                self.agent.front_depth_camera.intrinsics_matrix @ prev_2d @ previous_depth).T

    def get_rigid_transformation3d(self, A, B):
        """
        Find rigid body transformation between A and B
        :param A: 3 X n matrix of points
        :param B: 3 x n matrix of points
        :return: rigid body transformation between A and B
        code is from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
        solve the problem RA + t = B
        """

        assert len(A) == len(B)
        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))
        [num_rows, num_cols] = B.shape
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # subtract mean
        Am = A - np.tile(centroid_A, (1, num_cols))
        Bm = B - np.tile(centroid_B, (1, num_cols))

        # compute covariance matrix
        H = Am * np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        # special reflection case, when the determinant is -1
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T * U.T

        t = -R * centroid_A + centroid_B

        return R, t

    def estimate_rigid_transformation(self, cur_p3d, prev_p3d, num=10, delta=30, inlier_bound=10):
        """
        find the best R, T between cur_p3d and prev_p3d
        :param cur_p3d: n X 3 matrix of 3d points in the current gray image
        :param prev_p3d: n X 3 matrix of 3d points in the previous gray image
        :param num: max number of iterations
        :param delta: bound used to check the quality of tranformation
        :param accuracy: the minimum number of inliers required
        :return: best rigid transformation R, T
        """

        rotation = None
        translation = None
        assert len(cur_p3d) == len(prev_p3d)

        accuracy = 0
        best_inliers = None
        for i in range(num):
            # create 3 random indices of the correspondence points
            idx = np.random.randint(len(cur_p3d), size=3)

            # make sure that A, B are wrapped with mat
            A = np.mat(prev_p3d[idx].T)
            B = np.mat(cur_p3d[idx].T)

            R, t = self.get_rigid_transformation3d(A, B)
            error = np.linalg.norm(np.matmul(R, prev_p3d.T) + t - cur_p3d.T, axis=0)

            inliers = error < delta
            acc = np.sum(inliers)

            if acc > accuracy and acc >= inlier_bound:
                accuracy = acc
                best_inliers = inliers

        if best_inliers is None:
            return None, None
        else:
            idx = np.where(best_inliers > 0)[0]
            rotation, translation = self.get_rigid_transformation3d(np.mat(prev_p3d.T[:, idx]),
                                                                    np.mat(cur_p3d.T[:, idx]))

        return rotation, translation

    def shutdown(self):
        self.running = False
