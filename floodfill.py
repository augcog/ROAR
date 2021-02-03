import numpy as np
import cv2
from cv2 import connectedComponentsWithStats as getConnects
import pyrealsense2 as rs
import time

# Setup
pipeline = rs.pipeline()
pc = rs.pointcloud()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

# Setting up filters
# dec_filter = rs.decimation_filter()
# dec_filter.set_option(rs.option.filter_magnitude, 4)

# spa_filter = rs.spatial_filter()
# spa_filter.set_option(rs.option.filter_magnitude, 5)
# spa_filter.set_option(rs.option.filter_smooth_alpha, 1)
# spa_filter.set_option(rs.option.filter_smooth_delta, 50)
# spa_filter.set_option(rs.option.holes_fill, 3)

# tmp_filter = rs.temporal_filter()
# hol_fillin = rs.hole_filling_filter()

# dep_to_dis = rs.disparity_transform(True)
# dis_to_dep = rs.disparity_transform(False)

# Setting up indices
horizon_row = 200
d1, d2 = int(480 / 1), int(640 / 1)
idx, jdx = np.indices((d1, d2))
idx, jdx = idx[horizon_row:, :], jdx[horizon_row:, :]
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
# norm_fill = np.zeros((idx.shape[0]))


def depth_filter(aligned_depth_frame):
    # aligned_depth_frame = dec_filter.process(aligned_depth_frame)
    # aligned_depth_frame = dep_to_dis.process(aligned_depth_frame)
    # aligned_depth_frame = spa_filter.process(aligned_depth_frame)
    # aligned_depth_frame = tmp_filter.process(aligned_depth_frame)
    # aligned_depth_frame = dis_to_dep.process(aligned_depth_frame)
    # aligned_depth_frame = hol_fillin.process(aligned_depth_frame)
    return aligned_depth_frame


def normalize_v3(arr):
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    lens[lens <= 0] = 1
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def start():
    freeze = False

    while True:
        t1 = time.time()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_frame = depth_filter(aligned_depth_frame)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Apply flood fill
        points = pc.calculate(aligned_depth_frame)

        vtx = np.ndarray(
            buffer=points.get_vertices(), dtype=np.float32, shape=(d1 * d2, 3)
        )
        x = vtx[f1, :] - vtx[f2, :]
        y = vtx[f3, :] - vtx[f4, :]
        xyz_norm = normalize_v3(np.cross(x, y))

        # OpenCV FloodFill
        curr_img = xyz_norm.reshape((d1 - horizon_row, d2, 3)).astype(np.float32)
        mask = np.zeros((curr_img.shape[0] + 2, curr_img.shape[1] + 2), np.uint8)
        seed_point = (curr_img.shape[1] // 2, curr_img.shape[0] - 25)  # (d1 - 2, int(d2 / 2))
        # print(color_image.shape, curr_img.shape, mask.shape, seed_point)

        _, retval, _, _ = cv2.floodFill(image=curr_img,
                                        seedPoint=seed_point,
                                        newVal=(0, 0, 0),
                                        loDiff=(0.25, 0.25, 0.25),
                                        upDiff=(0.25, 0.25, 0.25),
                                        mask=mask)

        norm_umatrix = np.zeros((d1, d2))
        norm_umatrix[horizon_row:, :] = mask[1:-1, 1:-1] * 255
        # norm_umatrix = mask[1:-1, 1:-1] * 255

        # cv2.resize(cv2.UMat(mask[1:-1, 1:-1] * 255), (d2 * 1, d1 * 1))
        # print(norm_umatrix.shape)
        # cv2.imshow("Mask", norm_umatrix)
        # conv_depth = depth_image / np.max(depth_image) * 255
        # conv_depth = conv_depth.astype(np.uint8)
        # cv2.imshow("Depth", conv_depth)

        # comps, out, stats, cents = getConnects(norm_umatrix, connectivity=4)
        # sizes = stats.get()[:, 2] # get the area sizes
        # max_label = np.argmax(sizes[1:comps]) + 0
        color_image[norm_umatrix > 0] = 255  # [out.get() != max_label] = 255
        t2 = time.time()

        text = f'FPS: {1 / (t2 - t1)}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        place = (50, 50)
        thicc = 2

        color_image = cv2.putText(
            color_image, text, place, font, 1, color, thicc, cv2.LINE_AA
        )
        # seed_point = (seed_point[0], seed_point[1] + horizon_row)
        # print(seed_point)
        # color_image = cv2.circle(color_image, seed_point, 6, (255,0,0), 2)
        cv2.imshow('Color', color_image)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()


start()

#
# import numpy as np
# import cv2
# import pathlib
# import os
#
# dir_path = pathlib.Path(__file__).parent.absolute()
# h = 480
# w = 640
# DEBUG = True
# normal_window_size = 5
# normal_window_half_size = normal_window_size // 2
# normal_window_step = 5
#
#
# def computer_normal(normal_window):
#     normal = np.zeros((3,))
#     # Extract only nonzero normals from the normal_window
#     wh, ww, _ = normal_window.shape
#     point_list = []
#     for h_index in range(wh):
#         for w_index in range(ww):
#             if normal_window[h_index, w_index, 2] != 0:
#                 point_list.append(normal_window[h_index, w_index, :])
#
#     if len(point_list) > 3:
#         vector_list = []
#         for index1 in range(len(point_list) - 1):
#             for index2 in range(index1 + 1, len(point_list)):
#                 vector_list.append(point_list[index1] - point_list[index2])
#         normal_array = np.vstack(vector_list)
#         U, S, Vh = np.linalg.svd(normal_array)
#         normal = Vh[-1, :]
#
#         # Normal may point to opposite directions
#         # For ground-plane detection, we force positive Y direction
#         if normal[1] < 0:
#             normal = -normal
#
#     return normal
#
#
# point_cloud_paths = sorted(pathlib.Path(dir_path / "data/pointcloud_realsense/pointcloud").iterdir(),
#                            key=os.path.getmtime)
# rgb_paths = sorted(pathlib.Path(dir_path / "data/pointcloud_realsense/rgb").iterdir(), key=os.path.getmtime)
# counter = 1
# # Range = range(min(len(rgb_paths), len(point_cloud_paths)))
# Range = [1]
# for i in Range:
#     pointcloud_path = point_cloud_paths[i]
#     rgb_path = rgb_paths[i]
#     image = cv2.imread(rgb_path.as_posix())
#
#     points = np.load(pointcloud_path.as_posix())
#     xyz_map = points.reshape((h, w, 3))
#
#     if DEBUG:
#         depth_map = xyz_map[:, :, 2]  # invalid points will have depth = 0
#         # print(depth_map)
#         # Rescale for display
#         depth_map = depth_map / 10.0
#         cv2.imshow("Original RGB", image)
#         cv2.imshow("Depth", depth_map)
#
#     # Set empty normal array
#     normal_map = np.zeros((h // normal_window_step, w // normal_window_step, 3))
#     for h_index in range(0, h, normal_window_step):
#         for w_index in range(0, w, normal_window_step):
#             # Compute a consensus normal vector within the normal window, check whether window is out of bound
#             normal_window = xyz_map[
#                             max(0, h_index - normal_window_half_size): min(h - 1, h_index + normal_window_half_size),
#                             max(0, w_index - normal_window_half_size): min(w - 1, w_index + normal_window_half_size), :]
#             normal_map[h_index // normal_window_step, w_index // normal_window_step, :] = computer_normal(normal_window)
#
#     if DEBUG:
#         Y_map = normal_map[:, :, 1]
#         Y_map = cv2.normalize(Y_map, None, 1, 0, cv2.NORM_MINMAX)
#         cv2.imshow("Normal", Y_map)
#
#     # Select the seed point from -10 row
#     Y_array = normal_map[-10, :, 1]
#     # Select the point with median Y value pointing perpendicular to the ground
#     seed_w = np.argsort(Y_array)[len(Y_array) // 2]
#     seed_h = h // normal_window_step - 10
#
#     # Floodfill the normal array
#     seed_point = (seed_w, seed_h)  # Note opencv coordinates and numpy are reversed
#     mask = np.zeros((normal_map.shape[0] + 2, normal_map.shape[1] + 2), np.uint8)
#     normal_map = normal_map.astype('float32')
#     cv2.floodFill(normal_map, mask, seed_point, (128, 128, 128),
#                   loDiff=(0.15, 0.15, 0.15), upDiff=(0.15, 0.15, 0.15), flags=8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
#
#     if DEBUG:
#         cv2.imshow("Ground Plane Mask", mask)
#         cv2.waitKey(0)