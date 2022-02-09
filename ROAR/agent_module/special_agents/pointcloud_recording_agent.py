import logging

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData, Transform, Location
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
from enum import Enum
from typing import Optional, List, Tuple
from collections import deque
from ROAR.control_module.real_world_image_based_pid_controller import RealWorldImageBasedPIDController as ImageBasedPIDController
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import requests
import open3d as o3d
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector


class PointcloudRecordingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.prev_steerings: deque = deque(maxlen=10)
        self.agent_settings.pid_config_file_path = (Path(self.agent_settings.pid_config_file_path).parent /
                                                    "iOS_pid_config.json").as_posix()
        self.controller = ImageBasedPIDController(agent=self)

        # START LOC
        self.start_loc: Optional[Transform] = None
        self.start_loc_bound: float = 0.2
        self.has_exited_start_loc: bool = False

        # STOP Mid step
        self.ip_addr = "10.0.0.2"

        # Waypoint Following
        self.waypoints: List[Transform] = []
        self.curr_waypoint_index = 0
        self.closeness_threshold = 0.4

        # occupancy grid map
        # point cloud visualization
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(width=500, height=500)
        # self.pcd = o3d.geometry.PointCloud()
        # self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # self.points_added = False

        # pointcloud and ground plane detection
        self.depth2pointcloud = DepthToPointCloudDetector(agent=self)
        self.max_dist = 1.5
        self.height_threshold = 1
        self.ransac_dist_threshold = 0.01
        self.ransac_n = 3
        self.ransac_itr = 100

        self.waypoint_map: Optional[Map] = None
        buffer = 10
        x_scale = 20
        y_scale = 20
        x_offset = 100
        y_offset = 100
        self.occu_map = Map(
            x_offset=x_offset, y_offset=y_offset, x_scale=x_scale, y_scale=y_scale,
            x_width=2500, y_height=2500, buffer=10, name="occupancy map"
        )
        self.m = np.zeros(shape=(self.occu_map.map.shape[0], self.occu_map.map.shape[1], 3))

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(PointcloudRecordingAgent, self).run_step(sensors_data, vehicle)
        if self.front_rgb_camera.data is not None and self.front_depth_camera.data is not None:
            self.prev_steerings.append(self.vehicle.control.steering)
            try:
                pcd: o3d.geometry.PointCloud = self.depth2pointcloud.run_in_series(self.front_depth_camera.data,
                                                                                   self.front_rgb_camera.data)
                folder_name = Path("./data/pointcloud")
                folder_name.mkdir(parents=True, exist_ok=True)
                o3d.io.write_point_cloud((folder_name / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')}.pcd").as_posix(),
                                         pcd, print_progress=True)

                pcd = self.filter_ground(pcd)

                points = np.asarray(pcd.points)
                new_points = np.copy(points)

                points = np.vstack([new_points[:, 0], new_points[:, 2]]).T

                self.occu_map.update(points, val=1)
                coord = self.occu_map.world_loc_to_occu_map_coord(loc=self.vehicle.transform.location)
                self.m[np.where(self.occu_map.map == 1)] = [255, 255, 255]
                self.m[coord[1] - 2:coord[1] + 2, coord[0] - 2:coord[0] + 2] = [0, 0, 255]
                cv2.imshow("m", self.m)
            except Exception as e:
                print(e)

        return VehicleControl()

    @staticmethod
    def load_data(file_path: str) -> List[Transform]:
        waypoints = []
        f = Path(file_path).open('r')
        for line in f.readlines():
            x, y, z = line.split(",")
            x, y, z = float(x), float(y), float(z)
            l = Location(x=x, y=y, z=z)
            waypoints.append(Transform(location=l))
        return waypoints

    def filter_ground(self, pcd: o3d.geometry.PointCloud, max_dist: float = -1, height_threshold=0.1,
                      ransac_dist_threshold=0.01, ransac_n=3, ransac_itr=100) -> o3d.geometry.PointCloud:
        """
        Find ground from point cloud by first selecting points that are below the (car's position + a certain threshold)
        Then it will take only the points that are less than `max_dist` distance away
        Then do RANSAC on the resulting point cloud.

        Note that this function assumes that the ground will be the largest plane seen after filtering out everything
        above the vehicle

        Args:
            pcd: point cloud to be parsed
            max_dist: maximum distance
            height_threshold: additional height padding
            ransac_dist_threshold: RANSAC distance threshold
            ransac_n: RANSAC starting number of points
            ransac_itr: RANSAC number of iterations

        Returns:
            point cloud that only has the ground.

        """

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # height and distance filter
        # 0 -> left and right | 1 -> up and down | 2 = close and far
        points_of_interest = np.where((points[:, 1] < 0.3))
        points = points[points_of_interest]
        colors = colors[points_of_interest]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_dist_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_itr)

        pcd: o3d.geometry.PointCloud = pcd.select_by_index(inliers)
        pcd = pcd.voxel_down_sample(0.01)
        return pcd

    def waypoint_visualize(self,
                           map_data: np.ndarray,
                           name: str = "waypoint_visualization",
                           car_location: Optional[Location] = None,
                           next_waypoint_location: Optional[Location] = None):
        m = np.zeros(shape=(map_data.shape[0], map_data.shape[1], 3))
        m[np.where(map_data > 0.9)] = [255, 255, 255]
        point_size = 2
        if car_location is not None:
            coord = self.waypoint_map.world_loc_to_occu_map_coord(car_location)
            m[coord[1] - point_size:coord[1] + point_size, coord[0] - point_size:coord[0] + point_size] = [0, 0, 255]

        if next_waypoint_location is not None:
            coord = self.waypoint_map.world_loc_to_occu_map_coord(next_waypoint_location)
            m[coord[1] - point_size:coord[1] + point_size, coord[0] - point_size:coord[0] + point_size] = [0, 255, 0]
        cv2.imshow(name, m)
        cv2.waitKey(1)


    """
    Lane Following
    """

    def find_error(self):
        # make rgb and depth into the same shape
        data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                      dsize=(192, 256))
        # cv2.imshow("rgb_mask", cv2.inRange(data, self.rgb_lower_range, self.rgb_upper_range))
        data = self.rgb2ycbcr(data)
        # cv2.imshow("ycbcr_mask", cv2.inRange(data, self.ycbcr_lower_range, self.ycbcr_upper_range))
        # find the lane
        error_at_10 = self.find_error_at(data=data,
                                         y_offset=10,
                                         error_scaling=[
                                             (20, 0.1),
                                             (40, 0.75),
                                             (60, 0.8),
                                             (80, 0.9),
                                             (100, 0.95),
                                             (200, 1)
                                         ])
        error_at_50 = self.find_error_at(data=data,
                                         y_offset=50,
                                         error_scaling=[
                                             (20, 0.2),
                                             (40, 0.4),
                                             (60, 0.7),
                                             (70, 0.7),
                                             (80, 0.7),
                                             (100, 0.8),
                                             (200, 0.8)
                                         ]
                                         )

        if error_at_10 is None and error_at_50 is None:
            return None

        # we only want to follow the furthest thing we see.
        error = 0
        if error_at_10 is not None:
            error = error_at_10
        if error_at_50 is not None:
            error = error_at_50
        return error

    def find_error_at(self, data, y_offset, error_scaling) -> Optional[float]:
        y = data.shape[0] - y_offset
        lane_x = []
        cv2.imshow("data", data)
        # mask_red = cv2.inRange(src=data, lowerb=(0, 150, 60), upperb=(250, 240, 140))  # TERRACE RED
        # mask_yellow = cv2.inRange(src=data, lowerb=(0, 130, 0), upperb=(250, 200, 110)) # TERRACE YELLOW
        # mask_red = cv2.inRange(src=data, lowerb=(0, 180, 60), upperb=(250, 240, 140))  # CORY 337 RED
        # mask_yellow = cv2.inRange(src=data, lowerb=(0, 140, 0), upperb=(250, 200, 80))  # CORY 337 YELLOW
        mask_blue = cv2.inRange(src=data, lowerb=(60, 70, 120), upperb=(170, 130, 255))  # SHUWEI BLUE
        mask = mask_blue
        # mask = mask_red | mask_yellow

        # cv2.imshow("Lane Mask (Red)", mask_red)
        # cv2.imshow("Lane Mask (Yellow)", mask_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("Lane Mask (mask_blue)", mask)

        for x in range(0, data.shape[1], 5):
            if mask[y][x] > 0:
                lane_x.append(x)

        if len(lane_x) == 0:
            return None

        # if lane is found
        avg_x = int(np.average(lane_x))

        # find error
        center_x = data.shape[1] // 2

        error = avg_x - center_x
        # we want small error to be almost ignored, only big errors matter.
        for e, scale in error_scaling:
            if abs(error) <= e:
                # print(f"Error at {y_offset} -> {error, scale} -> {error * scale}")
                error = error * scale
                break

        return error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = self.controller.long_pid_control()
        # self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control

    def rgb2ycbcr(self, im):
        xform = np.array([[.299, .587, .114],
                          [-.1687, -.3313, .5],
                          [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)

    def no_line_seen(self):
        # did not see the line
        neutral = -90
        incline = self.vehicle.transform.rotation.pitch - neutral
        if incline < -10:
            # is down slope, execute previous command as-is
            # get the PID for downhill
            long_control = self.controller.long_pid_control()
            self.vehicle.control.throttle = long_control
            return self.vehicle.control
        else:
            # is flat or up slope, execute adjusted previous command
            return self.execute_prev_command()

    def non_blocking_pcd_visualization(self, pcd: o3d.geometry.PointCloud,
                                       should_center=False,
                                       should_show_axis=False,
                                       axis_size: float = 1):
        """
        Real time point cloud visualization.

        Args:
            pcd: point cloud to be visualized
            should_center: true to always center the point cloud
            should_show_axis: true to show axis
            axis_size: adjust axis size

        Returns:
            None

        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                          origin=np.mean(points,
                                                                                                         axis=0))
                self.vis.add_geometry(self.coordinate_frame)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if should_show_axis:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size,
                                                                                          origin=np.mean(points,
                                                                                                         axis=0))
                self.vis.update_geometry(self.coordinate_frame)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()


class Map:
    def __init__(self,
                 x_offset: float, y_offset: float, x_scale: float, y_scale: float,
                 x_width: int = 5000, y_height: int = 5000, buffer: int = 100,
                 name: str = "map"
                 ):
        self.name = name
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.x_width = x_width
        self.y_height = y_height
        self.buffer = buffer
        self.map = np.zeros(shape=(self.y_height, self.x_width))

    def world_loc_to_occu_map_coord(self, loc: Location) -> Tuple[int, int]:
        """
        Takes in a coordinate in the world reference frame and transform it into the occupancy map coordinate by
        applying the equation
        `int( (WORLD + OFFSET ) * SCALE)`

        Args:
            loc:

        Returns:

        """
        x = int((loc.x + self.x_offset) * self.x_scale) + self.buffer
        y = int((loc.z + self.y_offset) * self.y_scale) + self.buffer
        return x, y

    def world_arr_to_occu_map(self, arr: np.ndarray) -> np.ndarray:
        xs = ((arr[:, 0] + self.x_offset) * self.x_scale + self.buffer).astype(int)
        ys = ((arr[:, 1] + self.y_offset) * self.y_scale + self.buffer).astype(int)
        return np.array([xs, ys]).T

    def update(self, points: np.ndarray, val=1) -> int:
        """

        Args:
            val: value to update those points to
            points: points is a 2D numpy array consist of X and Z coordinates

        Returns:
            number of points updated
        """
        # print(np.min(points, axis=0), np.max(points, axis=0))

        points = self.world_arr_to_occu_map(points)
        self.map = np.zeros(shape=self.map.shape)
        self.map[points[:, 1], points[:, 0]] = val
        return len(points)

    def visualize(self, dsize: Optional[Tuple] = None):
        img = self.map.copy()
        if dsize:
            img = cv2.resize(img, dsize=dsize)
        cv2.imshow(self.name, img)

    @staticmethod
    def filter_outlier(track,
                       min_distance_btw_points: float = 0,
                       max_distance_btw_points: float = 0.2):
        filtered = []
        max_num_points_skipped = 0
        num_points_skipped = 0
        filtered.append(track[0])
        for i in range(1, len(track)):
            x2, z2 = track[i]
            x1, z1 = filtered[-1]
            diff_x, diff_z = abs(x2 - x1), abs(z2 - z1)
            if min_distance_btw_points < diff_x < max_distance_btw_points and min_distance_btw_points < diff_z < max_distance_btw_points:
                filtered.append([x2, z2])
                num_points_skipped = 0
            else:
                num_points_skipped += 1

            max_num_points_skipped = max(num_points_skipped, max_num_points_skipped)

        filtered = np.array(filtered)
        return filtered
