from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData, Transform
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
# from ROAR.control_module.simple_pid_controller import SimplePIDController
from ROAR.control_module.real_world_image_based_pid_controller import \
    RealWorldImageBasedPIDController as LaneFollowingPID
from collections import deque
from typing import List, Tuple, Optional
from ROAR.utilities_module.udp_multicast_communicator import UDPMulticastCommunicator
from ROAR.control_module.udp_pid_controller import UDP_PID_CONTROLLER
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import open3d as o3d
import math


class CS249Agent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.is_lead_car = False
        self.name = "car_2"
        self.car_to_follow = "car_1"

        self.udp_multicast = UDPMulticastCommunicator(agent=self,
                                                      mcast_group="224.1.1.1",
                                                      mcast_port=5004,
                                                      threaded=True,
                                                      update_interval=0.025,
                                                      name=self.name)
        self.add_threaded_module(self.udp_multicast)
        if self.is_lead_car:
            self.controller = LaneFollowingPID(agent=self)
        else:
            self.controller = UDP_PID_CONTROLLER(agent=self, distance_to_keep=1)
        self.prev_steerings: deque = deque(maxlen=10)

        # point cloud visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=500, height=500)
        self.pcd = o3d.geometry.PointCloud()
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.points_added = False

        # pointcloud and ground plane detection
        self.depth2pointcloud = DepthToPointCloudDetector(agent=self)
        self.max_dist = 1.5
        self.height_threshold = 0.5
        self.ransac_dist_threshold = 0.01
        self.ransac_n = 3
        self.ransac_itr = 100

        # occupancy map
        self.scaling_factor = 100
        self.occu_map = np.zeros(shape=(math.ceil(self.max_dist * self.scaling_factor),
                                        math.ceil(self.max_dist * self.scaling_factor)),
                                 dtype=np.float32)
        self.cx = len(self.occu_map) // 2
        self.cz = 0

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            pcd: o3d.geometry.PointCloud = self.depth2pointcloud.run_in_series(self.front_depth_camera.data,
                                                                               self.front_rgb_camera.data)
            pcd = self.filter_ground(pcd=pcd,
                                     max_dist=self.max_dist,
                                     height_threshold=self.height_threshold,
                                     ransac_dist_threshold=self.ransac_dist_threshold,
                                     ransac_n=self.ransac_n,
                                     ransac_itr=self.ransac_itr)

            self.non_blocking_pcd_visualization(pcd=pcd,
                                                axis_size=1,
                                                should_show_axis=True)
            occu_map = self.occu_map_from_pcd(pcd=pcd, scaling_factor=self.scaling_factor,
                                              cx=self.cx, cz=self.cz)
            left, center, right = self.find_obstacle_l_c_r(occu_map=occu_map, debug=True)

        return VehicleControl(brake=True)

        # if self.is_lead_car:
        #     return self.lead_car_step()
        # else:
        #     return self.follower_step()

    def find_obstacle_l_c_r(self, occu_map,
                                  left_occ_thresh=0.3,
                                  center_occ_thresh=0.5,
                                  right_occ_thresh=0.3,
                                  debug=False) -> Tuple[Tuple[bool, float], Tuple[bool, float], Tuple[bool, float]]:
        """
        Given an occupancy map `occu_map`, find whether in `left`, `center`, `right` which is/are occupied and
        also its probability for occupied.
        Args:
            occu_map: occupancy map
            left_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            center_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            right_occ_thresh: threshold occupied --> lower ==> more likely to be occupied
            debug: if true, draw out where the algorithm is looking at

        Returns:
            there tuples of bool and float representing occupied or not and its relative probability.
        """
        backtorgb = cv2.cvtColor(occu_map, cv2.COLOR_GRAY2RGB)

        height, width, channel = backtorgb.shape
        left_rec_start = (25 * width // 100, 40 * height // 100)
        left_rec_end = (80 * width // 100, 45 * height // 100)

        mid_rec_start = (25 * width // 100, 47 * height // 100)
        mid_rec_end = (80 * width // 100, 52 * height // 100)

        right_rec_start = (25 * width // 100, 55 * height // 100)
        right_rec_end = (80 * width // 100, 60 * height // 100)
        right = self.is_occupied(m=occu_map, start=right_rec_start, end=right_rec_end, threshold=left_occ_thresh)
        center = self.is_occupied(m=occu_map, start=mid_rec_start, end=mid_rec_end, threshold=center_occ_thresh)
        left = self.is_occupied(m=occu_map, start=left_rec_start, end=left_rec_end, threshold=right_occ_thresh)
        if debug:
            backtorgb = cv2.rectangle(backtorgb,
                                      left_rec_start,
                                      left_rec_end,
                                      (0, 0, 255), 1)

            backtorgb = cv2.rectangle(backtorgb,
                                      mid_rec_start,
                                      mid_rec_end,
                                      (0, 255, 0), 1)

            backtorgb = cv2.rectangle(backtorgb,
                                      right_rec_start,
                                      right_rec_end,
                                      (255, 0, 0), 1)

            cv2.imshow("show", backtorgb)
            cv2.waitKey(1)

        return left, center, right

    @staticmethod
    def is_occupied(m, start, end, threshold) -> Tuple[bool, float]:
        """
        Return the whether the area in `m` specified with `start` and `end` is occupied or not
        based on a ratio threshold.

        If the number of free spots / total area is less than threshold,
        then it means that this place is probability occupied.
        Args:
            m: 2D numpy array of occupancy map (free map to be exact)
            start: starting bounding box
            end: ending bounding box
            threshold: ratio to determine free or not

        Returns:
            bool -> true if occupied, false if free.

        """
        cropped = m[start[1]:end[1], start[0]:end[0]]
        area = (end[1] - start[1]) * (end[0] - start[0])
        spots_free = (cropped > 0.8).sum()
        ratio = spots_free / area
        return ratio < threshold,  ratio # if spots_free/area < threshold, then this place is occupied

    def occu_map_from_pcd(self, pcd: o3d.geometry.PointCloud, scaling_factor, cx, cz) -> np.ndarray:
        """
        Convert point cloud to occupancy map by first doing an affine transformation,
        then use the log odd update for updating the occupancy map

        Note that this method will update self.occu_map as well as returning the updated occupancy map.

        Args:
            pcd: point cloud
            scaling_factor: scaling factor for the affine transformation from pcd to occupancy map
            cx: x-axis constant for the affine transformation from pcd to occupancy map
            cz: z-axis constant for the affine transformation from pcd to occupancy map

        Returns:
            the updated occupancy map
        """
        points = np.asarray(pcd.points)
        points *= scaling_factor
        points = points.astype(int)
        points[:, 0] += cx
        points[:, 2] += cz
        np.clip(points, 0, len(self.occu_map))  # points that i see that are too far away
        self.occu_map -= 0.05
        self.occu_map[points[:, 0], points[:, 2]] += 0.9  # ground
        self.occu_map = self.occu_map.clip(0, 1)
        # kernel = np.ones((2, 2), np.uint8)
        # self.occu_map = cv2.morphologyEx(self.occu_map, cv2.MORPH_OPEN, kernel)  # erosion followed by dilation
        # self.occu_map = cv2.dilate(self.occu_map, kernel, iterations=2)  # to further filter out some noise
        return self.occu_map

    def filter_ground(self, pcd: o3d.geometry.PointCloud, max_dist: float = 2, height_threshold=0.5,
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
        points_of_interest = np.where((points[:, 1] < self.vehicle.transform.location.y + height_threshold) &
                                      (points[:, 2] < max_dist))
        points = points[points_of_interest]
        colors = colors[points_of_interest]
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_dist_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=ransac_itr)

        pcd = pcd.select_by_index(inliers)
        return pcd

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

    def lead_car_step(self):
        # if self.time_counter % 10 == 0:
        #     self.udp_multicast.send_current_state()
        if self.obstacle_found(debug=True):
            self.logger.info("Braking due to obstacle")
            return VehicleControl(brake=True)

        if self.is_light_found(debug=False):
            self.logger.info("Braking due to traffic light")
            return VehicleControl(brake=True)

        if self.front_rgb_camera.data is not None:
            error = self.find_error()
            if error is None:
                return self.no_line_seen()
            else:
                self.kwargs["lat_error"] = error
                self.vehicle.control = self.controller.run_in_series(next_waypoint=None)
                self.prev_steerings.append(self.vehicle.control.steering)
                return self.vehicle.control
        else:
            # image feed is not available yet
            return VehicleControl()

    def follower_step(self):
        if self.time_counter % 10 == 0:
            self.udp_multicast.send_current_state()
        # location x, y, z; rotation roll, pitch, yaw; velocity x, y, z; acceleration x, y, z
        if self.udp_multicast.msg_log.get(self.car_to_follow, None) is not None:
            control = self.controller.run_in_series(next_waypoint=Transform.from_array(
                self.udp_multicast.msg_log[self.car_to_follow]))
            return control
        else:
            # self.logger.info("No other cars found")
            return VehicleControl(throttle=0, steering=0)

    def is_light_found(self, low=(200, 200, 0), high=(255, 255, 100), n=500, debug=False) -> bool:
        """
        Find if there's light depending on if the image has n points that is within `low` and `high` range

        Args:
            n: minimum number of pixels to register as light found
            high: high range in the format of BGR
            low: low range in the format of BGR
            debug: True to show image

        Returns:
            True if light is found, False otherwise
        """
        if self.front_rgb_camera.data is not None:
            img = self.front_rgb_camera.data
            mask = cv2.inRange(img, low, high)
            if debug:
                cv2.imshow("traffic light", mask)
                print(sum(mask > 0))
            if sum(mask > 0) > n:
                return True
            return False
        else:
            return False

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

    def obstacle_found(self, threshold=0.468, debug=False) -> bool:
        """
        find obstacle by interpreting the depth image directly -- if in an area of interest, the area's average depth
        is smaller than threshold, then it means that there's probably something that is blocking the view and thus
        register as obstacle

        Args:
            threshold: minimum threshold to detect obstacle
            debug: true to show image

        Returns:
            True if obstacle is detected, false otherwise

        """
        if self.front_depth_camera.data is not None:
            depth_data = self.front_depth_camera.data
            roi = depth_data[70 * depth_data.shape[0] // 100: 90 * depth_data.shape[0] // 100,
                  30 * depth_data.shape[1] // 100: 60 * depth_data.shape[1] // 100]

            dist = np.average(roi)
            if debug:
                cv2.imshow("roi", roi)
                # cv2.imshow("depth", depth_data)
                print("distance to obstacle avg = ", dist)
            return dist < threshold
        else:
            return False

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
        mask_red = cv2.inRange(src=data, lowerb=(0, 180, 60), upperb=(250, 240, 140))  # CORY 337 RED
        mask_yellow = cv2.inRange(src=data, lowerb=(0, 140, 0), upperb=(250, 200, 80))  # CORY 337 YELLOW
        # mask = mask_yellow
        mask = mask_red | mask_yellow

        # cv2.imshow("mask_red", mask_red)
        # cv2.imshow("mask_yellow", mask_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("mask", mask)

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
