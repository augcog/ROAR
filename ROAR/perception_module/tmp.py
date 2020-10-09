import numpy as np
import cv2
from cv2 import connectedComponentsWithStats as getConnects
import pyrealsense2 as rs
import time
import open3d as o3d

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
dec_filter = rs.decimation_filter()
dec_filter.set_option(rs.option.filter_magnitude, 4)

spa_filter = rs.spatial_filter()
spa_filter.set_option(rs.option.filter_magnitude, 5)
spa_filter.set_option(rs.option.filter_smooth_alpha, 1)
spa_filter.set_option(rs.option.filter_smooth_delta, 50)
spa_filter.set_option(rs.option.holes_fill, 3)

tmp_filter = rs.temporal_filter()
hol_fillin = rs.hole_filling_filter()

dep_to_dis = rs.disparity_transform(True)
dis_to_dep = rs.disparity_transform(False)

# Setting up indices
d1, d2 = int(480 / 4), int(640 / 4)
idx, jdx = np.indices((d1, d2))
idx_back = np.clip(idx - 1, 0, idx.max()).flatten()
idx_front = np.clip(idx + 1, 0, idx.max()).flatten()
jdx_back = np.clip(jdx - 1, 0, jdx.max()).flatten()
jdx_front = np.clip(jdx + 1, 0, jdx.max()).flatten()
idx = idx.flatten()
jdx = jdx.flatten()

# rand_idx = np.random.choice(np.arange(idx.shape[0]), size=d1*d2, replace=False)
f1 = (idx_front * d2 + jdx)  # [rand_idx]
f2 = (idx_back * d2 + jdx)  # [rand_idx]
f3 = (idx * d2 + jdx_front)  # [rand_idx]
f4 = (idx * d2 + jdx_back)  # [rand_idx]
norm_fill = np.zeros((idx.shape[0]))

o3d_pointcloud = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()

def depth_filter(aligned_depth_frame):
    # aligned_depth_frame = dec_filter.process(aligned_depth_frame)
    aligned_depth_frame = dep_to_dis.process(aligned_depth_frame)
    aligned_depth_frame = spa_filter.process(aligned_depth_frame)
    aligned_depth_frame = tmp_filter.process(aligned_depth_frame)
    aligned_depth_frame = dis_to_dep.process(aligned_depth_frame)
    aligned_depth_frame = hol_fillin.process(aligned_depth_frame)
    return aligned_depth_frame


# TODO: get from calibration
# norm = np.array([ 0.01365045,  0.9828349 , -0.18398166], dtype=np.float32)
norm = np.array([-0.00994087, -0.99953604, 0.02879056], dtype=np.float32)


# norm = np.array([ 0.0037688 , -0.99291444,  0.11877182], dtype=np.float32)
# norm = np.array([-0.00919379, -0.99576974,  0.09142288], dtype=np.float32)

def normalize_v3(arr):
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    lens[lens <= 0] = 1
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


thresh = cv2.UMat(np.ones((d1 * 4, d2 * 4)) * 0.97)


def start():
    freeze = False
    counter = 0
    while True:
        try:
            t1 = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_depth_frame = depth_filter(aligned_depth_frame)
            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # Apply flood fill
            points = pc.calculate(aligned_depth_frame)
            vtx = np.ndarray(
                buffer=points.get_vertices(), dtype=np.float32, shape=(d1 * d2, 3)
            )
            o3d_pointcloud.points = o3d.utility.Vector3dVector(vtx)
            if counter == 0:
                vis.create_window(window_name="Open3d", width=400, height=400)
                vis.add_geometry(o3d_pointcloud)
            else:
                vis.update_geometry(o3d_pointcloud)
                vis.poll_events()
                vis.update_renderer()

            x = vtx[f1, :] - vtx[f2, :]
            y = vtx[f3, :] - vtx[f4, :]
            xyz_norm = normalize_v3(np.cross(x, y))
            norm_flat = xyz_norm @ norm
            norm_fill = norm_flat  # norm_fill[rand_idx] = norm_flat
            norm_matrix = np.abs(norm_fill.reshape((d1, d2)))

            norm_umatrix = cv2.resize(cv2.UMat(norm_matrix), (d2 * 4, d1 * 4))
            bool_matrix = cv2.compare(norm_umatrix, 0.95, cmpop=cv2.CMP_GT)
            comps, out, stats, cents = getConnects(bool_matrix, connectivity=4)
            sizes = stats.get()[:, 2]  # get the area sizes
            max_label = np.argmax(sizes[2:comps]) + 2
            color_image[out.get() == max_label] = 255
            t2 = time.time()

            text = f'FPS: {1 / (t2 - t1)}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 0, 0)
            place = (50, 50)
            thicc = 2

            color_image = cv2.putText(
                color_image, text, place, font, 1, color, thicc, cv2.LINE_AA
            )
            cv2.imshow('Color', color_image)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            counter += 1
        except:
            pass
    cv2.destroyAllWindows()
    pipeline.stop()


start()