from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.perception_module.legacy.ground_plane_point_cloud_detector import GroundPlanePointCloudDetector
from ROAR.visualization_module.visualizer import Visualizer
import numpy as np
import cv2
from pathlib import Path
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner

from typing import List
from ROAR.control_module.pid_controller import PIDParam
from ROAR.control_module.pid_controller import VehiclePIDController
from ROAR.utilities_module.data_structures_models import MapEntry


class PointCloudMapRecordingAgent(Agent):
    def __init__(self, **kwargs):
        super(PointCloudMapRecordingAgent, self).__init__(**kwargs)
        self.logger.debug("GPD2 Agent Initialized")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan
        self.controller = \
            self.pid_controller = VehiclePIDController(agent=self,
                                                       args_lateral=PIDParam.default_lateral_param(),
                                                       args_longitudinal=PIDParam.default_longitudinal_param(),
                                                       target_speed=20)
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        self.ground_plane_point_cloud_detector = GroundPlanePointCloudDetector(agent=self, max_points_to_convert=20000,
                                                                               ground_tilt_threshhold=0.05)
        self.visualizer = Visualizer(agent=self)
        self.map_history: List[MapEntry] = []
        self.file_written = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(PointCloudMapRecordingAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)
        control = self.local_planner.run_in_series()
        try:
            ground_points = self.ground_plane_point_cloud_detector.run_in_series()

            # print(np.shape(ground_points))
            color_image = self.front_rgb_camera.data.copy()
            ground_cords_in_2d: np.ndarray = self.visualizer.world_to_img_transform(xyz=ground_points)[:, :2]
            # this is a hack, without 5000 threshold, it sometimes have false detection
            # if np.shape(ground_cords_in_2d)[0] > 4000:
            # estimate left = (x_min, img_pos[1]) and right = (x_max, img_pos[1])
            img_positions = self.visualizer.world_to_img_transform(
                np.array([self.local_planner.way_points_queue[1].location.to_array()]))
            img_pos = img_positions[0]
            y_range = img_pos[1] - 5, img_pos[1] + 5
            indices = np.where(
                np.logical_and(ground_cords_in_2d[:, 1] >= y_range[0], ground_cords_in_2d[:, 1] <= y_range[1]))
            bar_cords = ground_cords_in_2d[indices]
            x_min, y_min = np.amin(bar_cords, axis=0)
            x_max, y_max = np.amax(bar_cords, axis=0)
            left_img_cord, right_img_cord = (x_min, img_pos[1]), (x_max, img_pos[1])
            pts = self.img_cords_to_world_cords(left_img_cord, right_img_cord)

            # save it
            self.map_history.append(MapEntry(point_a=pts[0].tolist(), point_b=pts[1].tolist()))

            # visualize
            color_image[ground_cords_in_2d[:, 1], ground_cords_in_2d[:, 0]] = [255, 255, 255]
            for y, x, _ in img_positions:
                color_image[x - 2: x + 2, y - 2:y + 2] = self.visualizer.GREEN
            image = cv2.line(color_image, left_img_cord, right_img_cord, (0, 255, 0), 5)
            cv2.imshow("color", image)
            cv2.waitKey(1)
        except Exception as e:
            self.logger.error(e)

        # write it to file
        if self.local_planner.is_done() and self.file_written is False:
            self.logger.debug("WRITING TO FILE")
            output_file_path: Path = Path(
                self.agent_settings.output_data_folder_path) / "easy_map_waypoints_pointcloud_v3.json"
            f = output_file_path.open('w')
            import json
            json.dump(fp=f, obj=[map_entry.dict() for map_entry in self.map_history], indent=2)
            f.close()
            self.file_written = True
        return control

    def img_cords_to_world_cords(self, left_img_cord, right_img_cord):
        """
        Converts depth data from the Front Depth Camera to World coordinates.

        Args:
            left_img_cord ():
            right_img_cord ():

        Returns:
            points: World coordinates in map
        """
        depth = self.front_depth_camera.data
        # depth_center = depth[img_pos_center[1]][img_pos_center[0]] * 1000
        depth_left = depth[left_img_cord[1]][left_img_cord[0]] * 1000
        depth_right = depth[right_img_cord[1]][right_img_cord[0]] * 1000

        # reconstruct p2d and transform it back to world space
        raw_p2d = np.array([
            [left_img_cord[0] * depth_left, left_img_cord[1] * depth_left, depth_left],
            # [right_img_cord[0] * depth_center, right_img_cord[1] * depth_center, depth_center],
            [right_img_cord[0] * depth_right, right_img_cord[1] * depth_right, depth_right]
        ])
        cords_y_minus_z_x = np.linalg.inv(self.front_depth_camera.intrinsics_matrix) @ raw_p2d.T
        cords_xyz_1 = np.vstack([
            cords_y_minus_z_x[2, :],
            cords_y_minus_z_x[0, :],
            -cords_y_minus_z_x[1, :],
            np.ones((1, np.shape(cords_y_minus_z_x)[1]))
        ])
        points: np.ndarray = self.vehicle.transform.get_matrix() @ self.front_depth_camera.transform.get_matrix() @ cords_xyz_1
        points = points.T[:, :3]
        return points

    @staticmethod
    def _pix2xyz(depth_img, i, j):
        return [
            depth_img[i, j] * j * 1000,
            depth_img[i, j] * i * 1000,
            depth_img[i, j] * 1000
        ]
