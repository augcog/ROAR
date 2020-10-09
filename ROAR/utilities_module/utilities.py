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


def img_to_world(depth_img,
                 intrinsics_matrix,
                 extrinsics_matrix,
                 sky_level=0.9,
                 depth_scaling_factor=1000) -> np.ndarray:
    # get a 2 x N array for their indices
    ground_loc = np.where(depth_img < sky_level)
    depth_val = depth_img[depth_img < sky_level] * depth_scaling_factor
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


def img_to_world2(depth_img,
                  intrinsics_matrix,
                  extrinsics_matrix,
                  segmentation:np.ndarray,
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
