from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.perception_module.legacy.semantic_segmentation_detector import GroundPlaneDetector
import cv2
import numpy as np
from ROAR.planning_module.local_planner .simple_waypoint_following_local_planner import SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.visualization_module.visualizer import Visualizer
from pathlib import Path
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from typing import List
from ROAR.utilities_module.data_structures_models import MapEntry


class MapGeneratingAgentV3(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ground_plane_detector = GroundPlaneDetector(agent=self)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PurePursuitController(
            agent=self,
            target_speed=40)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(vehicle=self.vehicle)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        self.visualizer = Visualizer(self)
        self.map_history: List[MapEntry] = []
        self.file_written = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(MapGeneratingAgentV3, self).run_step(sensors_data, vehicle)
        self.ground_plane_detector.run_in_series()
        control = self.local_planner.run_in_series()
        try:
            if self.ground_plane_detector.curr_segmentation is not None and \
                    len(self.ground_plane_detector.curr_segmentation) > 0:

                seg_visual = self.ground_plane_detector.curr_segmentation.copy()
                waypoint = self.local_planner.way_points_queue[2]
                img_pos_center = self.visualizer.calculate_img_pos(waypoint, camera=self.front_depth_camera)
                detection_line = seg_visual[img_pos_center[1], :, :]

                # find left obstacle position along detection line
                left_obstacle_pos = np.array([0, img_pos_center[1], 1])
                for x in range(img_pos_center[0], 0, -1):
                    if np.all(detection_line[x] == self.ground_plane_detector.OBSTACLE):
                        left_obstacle_pos[0] = x
                        break
                # find right obstacle position along detection line
                right_obstacle_pos = np.array([np.shape(self.front_depth_camera.data)[1] - 1, img_pos_center[1], 1])
                for x in range(img_pos_center[0], np.shape(self.front_depth_camera.data)[1] - 1, 1):
                    if np.all(detection_line[x] == self.ground_plane_detector.OBSTACLE):
                        right_obstacle_pos[0] = x
                        break
                # make visualization
                seg_visual[img_pos_center[1], :, :] = [0, 0, 255]
                seg_visual[left_obstacle_pos[1]:left_obstacle_pos[1] + 5,
                left_obstacle_pos[0]:left_obstacle_pos[0] + 5] = [0, 255, 0]

                seg_visual[img_pos_center[1] - 5:img_pos_center[1],
                img_pos_center[0] - 5:img_pos_center[0]] = [0, 255, 0]

                seg_visual[right_obstacle_pos[1] - 5:right_obstacle_pos[1],
                right_obstacle_pos[0] - 5:right_obstacle_pos[0]] = [0, 255, 0]

                cv2.imshow("seg_vis", seg_visual)
                cv2.waitKey(1)

                # find depth
                depth = self.front_depth_camera.data
                depth_center = depth[img_pos_center[1]][img_pos_center[0]] * 1000
                depth_left = depth[left_obstacle_pos[1]][left_obstacle_pos[0]] * 1000
                depth_right = depth[right_obstacle_pos[1]][right_obstacle_pos[0]] * 1000

                # reconstruct p2d and transform it back to world space
                raw_p2d = np.array([
                    [left_obstacle_pos[0] * depth_left, left_obstacle_pos[1] * depth_left, depth_left],
                    [img_pos_center[0] * depth_center, img_pos_center[1] * depth_center, depth_center],
                    [right_obstacle_pos[0] * depth_right, right_obstacle_pos[1] * depth_right, depth_right]
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

                # put it into the log
                map_entry = MapEntry(
                    point_a=points[0].tolist(),
                    point_b=points[2].tolist()
                )
                self.map_history.append(map_entry)
        except Exception as e:
            self.logger.error(f"Error during map making: {e}")

        if self.local_planner.is_done() and self.file_written is False:
            self.logger.debug("WRITING TO FILE")
            output_file_path: Path = Path(self.agent_settings.output_data_folder_path) / "easy_map_waypoints_v3.json"
            f = output_file_path.open('w')
            import json
            json.dump(fp=f, obj=[map_entry.dict() for map_entry in self.map_history], indent=2)
            f.close()
            self.file_written = True

        return control
