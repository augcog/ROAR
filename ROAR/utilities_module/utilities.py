import numpy as np


def png_to_depth(im: np.array) -> np.array:
    """
    Takes in an image read from cv2.imread(), whose output is simply a numpy
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

    """
    assert scaled_depth_image.shape[0] == 3, f"scaled depth image has incorrect shape [{scaled_depth_image.shape}]"
    assert intrinsics_matrix.shape == (3, 3), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"
    assert veh_world_matrix.shape == (4, 4), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"
    assert cam_veh_matrix.shape == (4, 4), f"Intrinsics matrix has incorrect shape [{intrinsics_matrix.shape}]"

    k_inv = np.linalg.inv(intrinsics_matrix)
    raw_p3d = k_inv @ scaled_depth_image
    raw_p3d_padded = np.vstack([raw_p3d, np.ones(shape=np.shape(raw_p3d)[1])])

    r_world_veh = np.linalg.inv(veh_world_matrix)
    r_veh_cam = np.linalg.inv(cam_veh_matrix)
    return (r_world_veh @ r_veh_cam @ raw_p3d_padded)[:3, :].T


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
