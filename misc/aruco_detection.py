import cv2
import cv2.aruco as aruco
import numpy as np
import os
from pathlib import Path
import math


def findArucoMarkers(img, arucoDict, arucoParam, intrinsics, distortion):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam,
                                               cameraMatrix=intrinsics,
                                               distCoeff=distortion)
    return bboxs, ids, rejected


def detectAruco(markerSize, totalMarkers, intrinsics: np.ndarray, distortion: np.ndarray, should_draw_axis=False):
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        if success:
            corners, ids, rejected = findArucoMarkers(img, arucoDict, arucoParam, intrinsics, distortion)
            if should_draw_axis and len(corners) > 0:
                for i in range(0, len(ids)):
                    # Estimate pose of each marker and return the values rvec and tvec---
                    #   (different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02,
                                                                                   intrinsics,
                                                                                   distortion)
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(img, corners)

                    # Draw Axis
                    cv2.aruco.drawAxis(img, intrinsics, distortion, rvec, tvec, 0.01)
                    # print(f"id = {ids[i]} --> tvec = {tvec}, rvec = {rvec}")
                    R = np.array(cv2.Rodrigues(rvec)[0])
                    P = constructTransformation(R, tvec)
                    euler_angles_rad = rotationMatrixToEulerAngles(R)
                    euler_angles_deg = np.rad2deg(euler_angles_rad)
                    print(f"roll: {euler_angles_deg[2]}, pitch: {euler_angles_deg[0]}, yaw: {euler_angles_deg[1]}")

                    # print(matrix)
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

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

    return np.array([x, y, z])


def constructTransformation(R, T):
    transformation_matrix = np.zeros([4, 4])
    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3] = T
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def loadCalib(caliberation_file: Path):
    npzfile = np.load(caliberation_file.as_posix())
    return npzfile['intrinsics'], npzfile['distortion'], \
           npzfile['new_intrinsics'], npzfile['roi']


def warpImageOnAruco(markerSize, totalMarkers, intrinsics, distortion, should_draw_axis):
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    cap = cv2.VideoCapture(0)
    pts_dst = np.array([[921, 731], [113, 732], [1127, 909], [927, 905]])  # caliberate this on first scene

    while True:
        success, img = cap.read()
        # img = cv2.flip(img, 1)
        if success:
            corners, ids, rejected = findArucoMarkers(img, arucoDict, arucoParam, intrinsics, distortion)
            if should_draw_axis and len(corners) > 0:
                for i in range(0, len(ids)):
                    # Estimate pose of each marker and return the values rvec and tvec---
                    #   (different from those of camera coefficients)
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02,
                                                                                   intrinsics,
                                                                                   distortion)
                    # Draw a square around the markers
                    cv2.aruco.drawDetectedMarkers(img, corners)

                    # Draw Axis
                    cv2.aruco.drawAxis(img, intrinsics, distortion, rvec, tvec, 0.01)

                    # print(corners[0][0])
                    h, status = cv2.findHomography(np.array(corners[0][0]), pts_dst)
                    im_out = cv2.warpPerspective(img, h, (img.shape[0], img.shape[1]))
                    cv2.imshow("imout", im_out)
                    # print(matrix)
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def main(markerSize, totalMarkers, intrinsics: np.ndarray, distortion: np.ndarray, should_draw_axis=False):
    detectAruco(markerSize, totalMarkers, intrinsics, distortion, should_draw_axis)
    # warpImageOnAruco(markerSize, totalMarkers, intrinsics, distortion, should_draw_axis)


if __name__ == '__main__':
    intrinsics, distortion, new_intrinsics, roi = loadCalib(Path("calib.npz"))
    # print(type(intrinsics), type(distortion))
    main(markerSize=5, totalMarkers=250, should_draw_axis=True,
         intrinsics=intrinsics, distortion=distortion)
