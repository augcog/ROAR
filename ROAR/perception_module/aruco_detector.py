from ROAR.perception_module.detector import Detector
from ROAR.agent_module.agent import Agent
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from ROAR.utilities_module.data_structures_models import Rotation


class ArucoDetector(Detector):
    def __init__(self, aruco_id: int, agent: Agent, aruco_dict_key='DICT_5X5_250', marker_length=0.1, **kwargs):
        super().__init__(agent, **kwargs)
        self.aruco_id: int = aruco_id
        self.aruco_dict = aruco.Dictionary_get(getattr(aruco, aruco_dict_key))
        self.aruco_param = aruco.DetectorParameters_create()
        self.marker_length = marker_length  # in meters

    def run_in_series(self, **kwargs):
        if self.agent.front_rgb_camera.data is not None:
            try:
                img = self.agent.front_rgb_camera.data.copy()
                result: dict = self.findArucoMarkers(img)  # {marker id -> bbox}
                # i only care about the aruco marker with id specified
                if result and self.aruco_id in result:
                    bbox = result[self.aruco_id]
                    # find R, T from aruco marker
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(bbox, self.marker_length,
                                                                                   self.agent.front_rgb_camera.intrinsics_matrix,
                                                                                   self.agent.front_rgb_camera.distortion_coefficient)
                    # convert rodrigues to 3x3 Rotation matrix
                    R = np.array(cv2.Rodrigues(rvec)[0])
                    T = tvec
                    # construct transformation matrix from R, and T
                    P = self.constructTransformation(R, T)
                    return P
            except Exception as e:
                return None
        return None

    def findArucoMarkers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxs, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_param,
                                                   cameraMatrix=self.agent.front_rgb_camera.intrinsics_matrix,
                                                   distCoeff=self.agent.front_rgb_camera.distortion_coefficient)
        log = dict()
        if ids is None:
            return log
        else:
            for i in ids:
                log[i[0]] = bboxs
            return log

    def save(self, **kwargs):
        pass

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R) -> Rotation:
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        euler_angles_deg = np.rad2deg(np.array([x, y, z]))

        rotation = Rotation(
            roll=euler_angles_deg[2],
            pitch=euler_angles_deg[0],
            yaw=euler_angles_deg[1],
        )
        return rotation

    @staticmethod
    def constructTransformation(R, T):
        transformation_matrix = np.zeros([4, 4])
        transformation_matrix[0:3, 0:3] = R
        transformation_matrix[0:3, 3] = T
        transformation_matrix[3, 3] = 1
        return transformation_matrix
