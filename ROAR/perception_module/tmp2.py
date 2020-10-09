import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import matplotlib.pyplot as plt  # 2D plotting library producing publication quality figures
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import open3d as o3d
import time

# Setup:
d1, d2 = 480, 640
idx, jdx = np.indices((d1, d2))
idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
idx = idx.flatten()
jdx = jdx.flatten()

f1 = (idx_front * d2 + jdx)  # [rand_idx]
f2 = (idx_back * d2 + jdx)  # [rand_idx]
f3 = (idx * d2 + jdx_front)  # [rand_idx]
f4 = (idx * d2 + jdx_back)  # [rand_idx]

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, d2, d1, rs.format.z16, 30)
config.enable_stream(rs.stream.color, d2, d1, rs.format.bgr8, 30)
profile = pipe.start(config)

norm = np.array([-0.00994087, -0.99953604, 0.02879056], dtype=np.float32)


def normalize_v3(arr):
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    lens[lens <= 0] = 1
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def construct_pointcloud(points) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    return pcd


# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()


counter = 0
while True:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    pc = rs.pointcloud()
    pc.map_to(color_frame)

    aligned_depth_frame = depth_frame
    pointcloud = pc.calculate(aligned_depth_frame)

    # starting from this point, the realsense code needs to
    # be exactly the same as the carla version
    vtx = np.ndarray(
        buffer=pointcloud.get_vertices(), dtype=np.float32, shape=(d1 * d2, 3)
    )

    t1 = time.time()
    x = vtx[f1, :] - vtx[f2, :]
    y = vtx[f3, :] - vtx[f4, :]
    xyz_norm = normalize_v3(np.cross(x, y))
    norm_flat = xyz_norm @ norm
    norm_matrix = norm_flat.reshape((d1, d2))

    bool_matrix = norm_matrix > 0.95
    color_image[bool_matrix] = 255

    t2 = time.time()
    print(t2 - t1)
    cv2.imshow("color", color_image)
    cv2.waitKey(1)
    counter += 1
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
pipe.stop()
