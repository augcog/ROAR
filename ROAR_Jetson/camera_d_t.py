import logging
import cv2
from cv2 import aruco
import pyrealsense2 as rs
import math as m
import numpy as np
import sys
import time
from typing import Optional, Tuple, List
try:
    from ROAR_Jetson.part import Part
except:
    from part import Part
CAM_CONFIG = {
    "image_w": 640,  # image width
    "image_h": 480,  # image height
    "framerate": 30,  # frame rate
    "aruco_dict": aruco.DICT_5X5_250,  # dictionary used in aruco
    "aruco_thres": 10,  # aruco marker threshold, internal parameter
    "aruco_block_size": 0.1592,  # length of aruco marker's side in meter
    "use_default_cam2cam": False,  # recommended true, no rotation is calculated
    "detect_mode": False,  # useful only in show_gui mode where a aruco-marker detection reference is given
    "calibrate_threshold": 1  # calibrate threshold in degree
}


class RealsenseD435iAndT265(Part):
    def __init__(self, image_w=CAM_CONFIG["image_w"], image_h=CAM_CONFIG["image_h"], framerate=CAM_CONFIG["framerate"],
                 show_gui=False) -> None:
        name = "Realsense D435i and T265"
        super().__init__(name=name)
        self.logger.debug("Initiating Intel Realsense")

        self.show_gui = show_gui
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        # Declare RealSense pipeline, encapsulating the actual device and sensors
        self.pipe_d = rs.pipeline()
        self.cfg_d = rs.config()
        self.cfg_d.enable_stream(rs.stream.color, image_w, image_h, rs.format.bgr8, framerate)  # color camera
        self.cfg_d.enable_stream(rs.stream.depth, image_w, image_h, rs.format.z16, framerate)  # depth camera
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # init for the t camera
        self.pipe_t = rs.pipeline()
        self.cfg_t = rs.config()
        self.cfg_t.enable_stream(rs.stream.pose)

        self.location: np.ndarray = np.array([0, 0, 0])  # x y z
        self.rotation: np.ndarray = np.array([0, 0, 0])  # pitch yaw roll
        self.velocity: np.ndarray = np.array([0, 0, 0])  # x y z
        self.acceleration: np.ndarray = np.array([0, 0, 0])  # x y z
        self.depth_camera_intrinsics: Optional[np.ndarray] = None
        self.rgb_camera_intrinsics: Optional[np.ndarray] = None
        self.depth_camera_distortion_coefficients: Optional[np.ndarray] = None
        self.rgb_camera_distortion_coefficients: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        self.depth_image: Optional[np.ndarray] = None

        # align_to = rs.stream.color
        # self.aligned_d = rs.align(rs.stream.color)

        # Start streaming with requested config
        self.prof_d, self.prof_t = None, None
        self.aligned_d = rs.align(rs.stream.color)
        try:

            self.prof_t = self.pipe_t.start(self.cfg_t)
            self.prof_d = self.pipe_d.start(self.cfg_d)
            self.logger.info("cam initialized")
        except Exception as e:
            raise ConnectionError(f"Error {e}. Pipeline Initialization Error")

        self.calibrated = False
        # setup all the color/depth frame intrinsics (distortion coefficients + camera matrix)
        self.set_intrinsics()

        # detection related params
        self.aruco_dict = aruco.Dictionary_get(CAM_CONFIG['aruco_dict'])
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = CAM_CONFIG['aruco_thres']
        self.block_size = CAM_CONFIG['aruco_block_size']
        self.use_default_cam2cam = CAM_CONFIG['use_default_cam2cam']
        self.detect_mode = CAM_CONFIG['detect_mode']
        self.calibrate_thres = CAM_CONFIG['calibrate_threshold']

        """
        t2d: transformation matrix from t-camera coordinate to d-camera coordinate
        d2m: transformation matrix from d-camera coordinate to marker coordinate
        t2m: transformation matrix from t-camera coordinate to marker coordinate
        """
        self.t2d, self.d2m, self.t2m = None, None, None

        self.logger.info("Camera Initiated")

    def run_step(self):
        self.poll()

    def shutdown(self):

        try:
            self.pipe_d.stop()
            self.logger.debug("D435i shut down success")
        except Exception as e:
            self.logger.error(e)
            raise ConnectionError(f"D435i unable to shutdown: {e}")
        try:
            self.pipe_t.stop()
            self.logger.debug("T265 shutdown success")
        except Exception as e:
            self.logger.error(e)
            raise ConnectionError(f"T265 unable to shutdown: {e}")


    def set_intrinsics(self):
        rgb_intr = self.prof_d.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.rgb_camera_intrinsics = np.array([[rgb_intr.fx, 0, rgb_intr.ppx],
                                               [0, rgb_intr.fy, rgb_intr.ppy],
                                               [0, 0, 1]])
        self.rgb_camera_distortion_coefficients = np.array(rgb_intr.coeffs)

        depth_intr = self.prof_d.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.depth_camera_intrinsics = np.array([[depth_intr.fx, 0, depth_intr.ppx],
                                                 [0, depth_intr.fy, depth_intr.ppy],
                                                 [0, 0, 1]])
        self.depth_camera_distortion_coefficients = np.array(depth_intr.coeffs)

    def get_intrinsics(self):
        return {
            'rgb_camera_intrinsics': self.rgb_camera_intrinsics,
            'rgb_camera_distortion_coefficients': self.rgb_camera_distortion_coefficients,
            'depth_camera_intrinsics': self.depth_camera_intrinsics,
            'depth_camera_distortion_coefficients': self.depth_camera_distortion_coefficients
        }

    def stop(self):
        self.pipe_d.stop()
        self.pipe_t.stop()
        self.logger.debug("Shutting Down")

    def restart(self, t_cam=False, d_cam=False):
        try:
            if t_cam:
                self.pipe_t.stop()
                self.prof_t = self.pipe_t.start(self.cfg_t)
            if d_cam:
                self.pipe_d.stop()
                self.prof_d = self.pipe_d.start(self.cfg_d)
        except Exception as e:
            raise ConnectionError(f"Error {e}. Pipeline Initialization Error")

    def start_detect(self):
        self.detect_mode = True

    def stop_detect(self):
        self.detect_mode = False

    def recaliberate(self,
                     uncaliberated_tvec: np.ndarray,
                     uncaliberated_rvec: np.ndarray):
        d2m, _, _ = self.get_trans_mat(self.color_image)
        is_aruco_marker_detected = d2m is not None  # true if aruco marker detected, false otherwise
        if is_aruco_marker_detected:
            self.d2m = d2m
            self.t2d = self.cam2cam(uncaliberated_tvec, uncaliberated_rvec)
            self.t2m = self.d2m @ self.t2d
            self.calibrated = True
        else:
            self.calibrated = False
            self.d2m, self.t2d, self.t2m = None, None, None

    def poll(self):
        """
        Attempt to poll from T265 and D435i Camera.
        Update data & return True if operation is a success, false otherwise

        Data includes
        color_image, depth_image, location, rotation, velocity

        Returns:
            True if operation is success, false otherwise
        """
        try:
            frame_d = self.pipe_d.wait_for_frames()
            aligned_frames = self.align.process(frame_d)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            self.color_image: np.ndarray = np.asanyarray(color_frame.get_data())
            self.depth_image: np.ndarray = np.asanyarray(depth_frame.get_data())

            frame_t = self.pipe_t.wait_for_frames()
            pose_frame = frame_t.get_pose_frame()
            t_tvec, t_rvec, t_vvec = self.pose_frame_data(pose_frame)
            if self.calibrated is False:
                self.logger.info("Warning: System is not caliberated")

                self.recaliberate(uncaliberated_tvec=t_tvec,
                                  uncaliberated_rvec=t_rvec)
            else:
                self.location, self.rotation, self.velocity = self.to_global_coord(t_tvec, t_rvec, t_vvec)
                # self.location, self.rotation, self.velocity = self.to_roar_coord(location, rotation, velocity)

            return True

        except Exception as e:
            logging.error(e)
            return False

    def to_global_coord(self,
                        t_tvec: np.ndarray,
                        t_rvec: np.ndarray,
                        t_vvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform coordiante from local to world coordinate frame
        Args:
            t_tvec: 4x1 array of [x,y,z,1] in local coordinate frame
            t_rvec: 4x1 array of [rx,ry,rz,w] in local coordinate frame
            t_vvec: 4x1 array of [vx,vy,vz,1] in local coordinate frame

        Returns:
            location: 3x1 array of [x,y,z] in global coordinate frame
            rotation: 3x1 array of [roll, pitch, yaw] in global coordinate frame
            velocity: 3x1 array of [vx, vy, vz] in global coordinate frame
        """
        if not self.calibrated:
            self.logger.error("Cannot convert to global coord because system is registered as uncaliberated")
            raise Exception("Uncaliberated Error")
        else:
            location = (self.t2m @ t_tvec)[:3]
            rotation = self.rvec_to_rpy(rvec=t_rvec)
            rotation = np.array([rotation[0], rotation[1], rotation[2]])
            velocity = np.array([t_vvec[0], t_vvec[1], t_vvec[2]])
            return location, rotation, velocity

    """
    returns the [x, y, z] global coordiante of the vehicle (camera) in the map
    default format is a 3x1 np array, return np.nan when the pipeline errors
    """

    def poll_global_loc(self, t_tvec, t_rvec):
        if not self.calibrated:
            self.logger.error("Camera not calibrated yet")
            return np.nan

        try:
            self.location = (self.t2m @ t_tvec)[:3]
            return self.location, t_rvec

        except Exception as e:
            logging.error(e)
            return np.nan

    @staticmethod
    def depth_frame_data(depth_frame):
        return np.asanyarray(depth_frame.get_data())

    @staticmethod
    def color_frame_data(color_frame):
        return np.asanyarray(color_frame.get_data())

    @staticmethod
    def pose_frame_data(pose_frame):
        data = pose_frame.get_pose_data()
        t = data.translation
        t_tvec = np.array([t.x, t.y, t.z, 1])
        r = data.rotation
        t_rvec = np.array([r.x, r.y, r.z, r.w])
        v = data.velocity
        t_vvec = np.array([v.x, v.y, v.z, 1])
        return t_tvec, t_rvec, t_vvec

    """
    Helper function that turns a rotation vector into roll, pitch, and yaw euler angles
    """

    @staticmethod
    def rvec_to_rpy(rvec) -> np.ndarray:
        x, y, z, w = tuple(rvec)
        w = w
        x, y, z = z, x, y

        pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi
        return np.array([roll, pitch, yaw])

    """
    This is a transformation from the d-camera's coord system to the marker's (world) system
    rvec and tvec are derived from the black-box algorithm in cv2.aruco, they represent some 
    important quantities from marker to d-camera. Since we want to extract the reverse transformation,
    we invert the matrix at the end.
    """

    @staticmethod
    def cam2marker(rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        trans_mat = np.identity(4)
        trans_mat[:3, :3] = rmat
        trans_mat[:3, 3] = tvec
        trans_mat = np.linalg.inv(trans_mat)
        return trans_mat

    """
    This is a transformation from the t-camera's coordinate system to d-camera's.
    ** Why do we need to do this since these cameras are installed together? ** 
    1) t's coordinate axes are aligned independent of its own physical rotation, while
    d's are dependent.
    2) there's still some minor translation between their coordinate systems, which will
    be implemented later :TODO @Star
    """

    def cam2cam(self, t_rvec, t_tvec):
        base_c2c = np.array([1, 0, 0, 0,
                             0, -1, 0, 0,
                             0, 0, -1, 0,
                             0, 0, 0, 1]).reshape((4, 4))
        # no tuning of the t camera rotation
        if self.use_default_cam2cam:
            return base_c2c
        else:
            trans_mat = self.rotation_adjust(t_rvec)
            return base_c2c @ trans_mat

    """
    output a transformation matrix that goes from the pose dependent coordinate system assumed by 
    ARUCO marker to the pose independent coordinate system (y pointing anti-gravity) that the 
    tracking module uses
    formula from https://dev.intelrealsense.com/docs/rs-trajectory
    """

    @staticmethod
    def rotation_adjust(r):
        mat = np.identity(4)
        mat[0, 0] = 1 - 2 * r[1] ** 2 - 2 * r[2] ** 2
        mat[0, 1] = 2 * r[0] * r[1] - 2 * r[2] * r[3]
        mat[0, 2] = 2 * r[0] * r[2] + 2 * r[1] * r[3]
        mat[1, 0] = 2 * r[0] * r[1] + 2 * r[2] * r[3]
        mat[1, 1] = 1 - 2 * r[0] ** 2 - 2 * r[2] ** 2
        mat[1, 2] = 2 * r[1] * r[2] - 2 * r[0] * r[3]
        mat[2, 0] = 2 * r[0] * r[2] - 2 * r[1] * r[3]
        mat[2, 1] = 2 * r[2] * r[1] + 2 * r[0] * r[3]
        mat[2, 2] = 1 - 2 * r[0] ** 2 - 2 * r[1]

        mat = np.linalg.inv(mat)
        return mat

    def get_trans_mat(self, img):
        corners, aruco_ids, _ = aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)
        if aruco_ids is None:
            return None, None, None
        else:
            for index in range(len(aruco_ids)):
                if aruco_ids[index][0] == 0:  # i am seeing aruco id=0
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], self.block_size,
                                                                    self.rgb_camera_intrinsics,
                                                                    self.rgb_camera_distortion_coefficients)
                    d2m = self.cam2marker(rvec, tvec)

                    if self.show_gui:
                        aruco.drawAxis(img, self.rgb_camera_intrinsics, self.rgb_camera_distortion_coefficients,
                                       rvec[0],
                                       tvec[0], 0.01)
                        aruco.drawDetectedMarkers(img, corners)

                    return d2m, tvec, rvec

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Takes in roll pitch yaw and compute rotation matrix using the order of

        R = R_yaw @ R_pitch @ R_roll

        http://planning.cs.uiuc.edu/node104.html

        Args:
            roll: float of roll in degree
            pitch: float of pitch in degree
            yaw: float of yaw in degree

        Returns:
            3 x 3 array rotation matrix
        """
        c_y = np.cos(np.radians(yaw))
        s_y = np.sin(np.radians(yaw))
        c_r = np.cos(np.radians(roll))
        s_r = np.sin(np.radians(roll))
        c_p = np.cos(np.radians(pitch))
        s_p = np.sin(np.radians(pitch))

        R_roll = np.array([
            [1, 0, 0],
            [0, c_r, -s_r],
            [0, s_r, c_r]
        ])
        R_pitch = np.array([
            [c_p, 0, s_p],
            [0, 1, 0],
            [-s_p, 0, c_p]
        ])
        R_yaw = np.array([
            [c_y, -s_y, 0],
            [s_y, c_y, 0],
            [0, 0, 1]
        ])
        return R_yaw @ R_pitch @ R_roll

    @staticmethod
    def rotation_matrix_to_euler(matrix: np.ndarray):
        r11, r21, r31, r32, r33 = matrix[0][0], matrix[1][0], matrix[2][0], matrix[2][1], matrix[2][2]
        yaw = alpha = np.degrees(np.arctan2(r21, r11))
        pitch = beta = np.degrees(np.arctan2(-r31, np.sqrt(r32 ** 2 + r33 ** 2)))
        roll = gamma = np.degrees(np.arctan2(r32, r33))
        return np.array([roll, pitch, yaw])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)
    camera = RealsenseD435iAndT265()
    # camera.calibrate(show_img=True)
    while True:
        camera.poll()
        img = camera.color_image.copy()
        cv2.putText(img=img, text=f"System is {'caliberated' if camera.calibrated else 'not caliberated'}",
                    org=(10, 25), fontFace=camera.font, fontScale=0.5,
                    color=(255, 255, 0), thickness=1, lineType=camera.line_type)
        cv2.putText(img=img, text=f"{camera.location}",
                    org=(10, 50), fontFace=camera.font, fontScale=0.5,
                    color=(255, 255, 0), thickness=1, lineType=camera.line_type)
        cv2.putText(img=img, text=f"{camera.rotation}",
                    org=(10, 75), fontFace=camera.font, fontScale=0.5,
                    color=(255, 255, 0), thickness=1, lineType=camera.line_type)
        cv2.putText(img=img, text=f"{camera.velocity}",
                    org=(10, 100), fontFace=camera.font, fontScale=0.5,
                    color=(255, 255, 0), thickness=1, lineType=camera.line_type)

        cv2.imshow("frame", img)
        key = cv2.waitKey(100)
        key_ord = key & 0xFF
        if key_ord == ord('q'):
            camera.stop()
            exit(1)
        elif key_ord == ord('r'):
            camera.calibrated = False
