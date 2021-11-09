import numpy as np
import socket


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def png_to_depth(im: np.array) -> np.array:
    """
    Takes in an image read from cv2.imread(), whose output_oct_10 is simply a numpy
    array,
    turn it into a depth image according to carla's method of
    (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    Args:
        im: input image, read from cv2.imread()
    Returns:
        depth image
    """
    im = im.astype(np.float64)
    normalized_depth = np.dot(im[:, :, :3], [1, 256, 65536.0])
    normalized_depth /= 16777215.0
    return normalized_depth


def img_to_world(scaled_depth_image: np.ndarray,
                 intrinsics_matrix: np.ndarray,
                 veh_world_matrix: np.ndarray,
                 cam_veh_matrix: np.ndarray) -> np.ndarray:
    """
    Compute image to world translation using the formula below

    ((R_world_veh)^(-1) @ (R_veh_cam)^(-1) @ ((intrinsics)^(-1) @ scaled_depth_image).pad_with_1)[:3, :] = [X Y Z]
    Args:
        scaled_depth_image: 3 x n numpy array
        intrinsics_matrix: 3 x 3 intrinsics
        veh_world_matrix: 4 x 4 vehicle to world transformation matrix
        cam_veh_matrix: 4 x 4 camera to vehicle transformation matrix

    Returns:
        n x 3 array of n points
    """
    assert scaled_depth_image.shape[0] == 3, f"scaled depth image has incorrect shape [{scaled_depth_image.shape}]"
    assert intrinsics_matrix.shape == (3, 3), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"
    assert veh_world_matrix.shape == (4, 4), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"
    assert cam_veh_matrix.shape == (4, 4), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"
    # extrinsics @ inv(K) @ [u, v,1] = [X,Y,Z]
    k_inv = np.linalg.inv(intrinsics_matrix)
    raw_p3d = k_inv @ scaled_depth_image

    ones = np.ones(shape=np.shape(raw_p3d)[1])
    # raw_p3d_padded = np.vstack([raw_p3d, ones])

    raw_p3d_padded = np.vstack([
        raw_p3d[2, :],
        raw_p3d[0, :],
        -raw_p3d[1, :],
        ones
    ])
    points: np.ndarray = (veh_world_matrix @ cam_veh_matrix @ raw_p3d_padded)[:3, :].T

    return points


def img_to_world2(depth_img,
                  intrinsics_matrix,
                  extrinsics_matrix,
                  segmentation: np.ndarray,
                  criteria,
                  depth_scaling_factor=1000) -> np.ndarray:
    # get a 2 x N array for their indices

    ground_loc = np.where(segmentation == criteria)[:2]
    # print(ground)
    # ground_loc = np.where(depth_img == criteria)
    depth_val = depth_img[ground_loc] * depth_scaling_factor
    ground_loc = ground_loc * depth_val

    # compute raw_points
    raw_points = np.vstack([ground_loc, depth_val])

    # convert to cords_y_minus_z_x
    cords_y_minus_z_x = np.linalg.inv(intrinsics_matrix) @ raw_points

    # convert to cords_xyz_1
    ones = np.ones((1, np.shape(cords_y_minus_z_x)[1]))

    cords_xyz_1 = np.vstack([
        cords_y_minus_z_x[2, :],
        cords_y_minus_z_x[0, :],
        -cords_y_minus_z_x[1, :],
        ones
    ])

    # multiply by cam_world_matrix
    points = extrinsics_matrix @ cords_xyz_1  # i have all points now
    return points


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Takes in roll pitch yaw and compute rotation matrix using the order of

    R = R_yaw * R_pitch * R_roll

    http://planning.cs.uiuc.edu/node104.html

    Args:
        roll: float of roll in degree
        pitch: float of pitch in degree
        yaw: float of yaw in degree

    Returns:
        3 x 3 array rotation matrix
    """
    ry, rx, rz = np.radians(yaw), np.radians(pitch), np.radians(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    return Rx @ Ry @ Rz

