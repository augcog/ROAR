import logging
from ROAR.utilities_module.data_structures_models import (
    Transform,
)
from ROAR.utilities_module.camera_models import Camera
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from typing import Optional, List

from deprecated import deprecated


class Visualizer:
    GREEN = [0, 255, 0]
    GROUND = [0, 0, 0]

    def __init__(self,
                 agent: Agent,
                 occupancy_grid_map: Optional[OccupancyGridMap] = None, ):
        self.logger = logging.getLogger(__name__)
        self.agent = agent
        self.occupancy_grid_map = occupancy_grid_map

    @deprecated(reason="Will no longer support seperate graph visualization")
    def visualize_waypoint(self, waypoint_transform: Transform):
        coords = self.world_to_img_transform(xyz=np.array([[waypoint_transform.location.x,
                                                           waypoint_transform.location.y,
                                                           waypoint_transform.location.z]]))
        img = self.agent.front_rgb_camera.data.copy()
        img = cv2.arrowedLine(img,
                              (400, 600),
                              (coords[0][0], coords[0][1]),
                              (0, 255, 0),
                              2)
        cv2.imshow("Next Waypoint", img)
        cv2.waitKey(1)

    @deprecated(reason="Will no longer support single image to world calculation.")
    def calculate_img_pos(self, waypoint_transform: Transform, camera: Camera) -> np.ndarray:
        """
        Calculate the 2D image coordinate from 3D world space
        Args:
            camera:
            waypoint_transform: Desired point in 3D world space
        Returns:
            Array if integers [X, Y, depth]
        """
        waypoint_location = waypoint_transform.location.to_array()  # [x, y, z]
        waypoint_location = np.concatenate(
            [waypoint_location, [1]]
        )  # 4 x 1 array [X, Y, Z, 1]
        veh_cam_matrix = camera.transform.get_matrix()  # 4 x 4
        world_veh_matrix = self.agent.vehicle.transform.get_matrix()  # 4 x 4

        world_cam_matrix = np.linalg.inv(np.dot(world_veh_matrix, veh_cam_matrix))

        cords_xyz = world_cam_matrix @ waypoint_location
        # print(cords_xyz)
        cords_y_minus_z_x = np.array([cords_xyz[1], -cords_xyz[2], cords_xyz[0]])
        # cords_y_minus_z_x = np.array([cords_xyz[0], cords_xyz[1], -cords_xyz[2]])
        raw_p2d = camera.intrinsics_matrix @ cords_y_minus_z_x
        print(camera.intrinsics_matrix)
        # print(raw_p2d)
        cam_cords = np.array(
            [raw_p2d[0] / raw_p2d[2], raw_p2d[1] / raw_p2d[2], raw_p2d[2]]
        )
        result = np.round(cam_cords, 0).astype(np.int64)
        # print(result)
        return result

    @deprecated(reason="Will no longer support next waypoint visualization on a single display")
    def visualize(self, next_waypoint_transform: Transform) -> None:
        """
        This function will allow multiple objects to be drawn on here.
        Args:
            next_waypoint_transform: Next Waypoint's Transform information
        Returns:
            None
        """
        next_waypoint_cam_pos = self.calculate_img_pos(
            waypoint_transform=next_waypoint_transform,
            camera=self.agent.front_depth_camera,
        )
        img = self.agent.front_rgb_camera.data.copy()

        start_point = (400, 600)

        img = cv2.arrowedLine(
            img=img,
            pt1=start_point,
            pt2=(next_waypoint_cam_pos[0], next_waypoint_cam_pos[1]),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imshow("Visualization", img)
        cv2.waitKey(1)

    @classmethod
    @deprecated(reason="will no longer support single semantic segmentation visualization")
    def visualize_semantic_segmentation(cls, semantic_segmetation) -> None:
        """
        Args:
            semantic_segmetation: Width x Height x 3 array
                                  with white = obstacles, black = ground,
                                  blue = sky
        Returns:
            None
        """

        if semantic_segmetation is not None:
            cv2.imshow("Semantic Segmentation", semantic_segmetation)
            cv2.waitKey(1)

    def world_to_img_transform(self, xyz: np.ndarray) -> np.ndarray:
        """
        Calculate the 2D image coordinate from 3D world space
        Args:
            xyz: (Nx3) array representing X, Y, Z in world coord
        Returns:
            Array if integers [u, v, f]
        """
        xyz1 = np.append(xyz, np.ones(shape=(len(xyz), 1)), axis=1)
        UVD = self.agent.front_depth_camera.intrinsics_matrix @ \
              self.agent.front_depth_camera.transform.get_matrix()[:3] @ \
              self.agent.vehicle.transform.get_matrix() @ xyz1.T
        uvd = np.array([
            UVD[0, :] / UVD[2, :], UVD[1, :] / UVD[2, :], UVD[2, :]
        ]).T
        result = np.round(uvd, 0).astype(np.int64)
        print(f"waypoint: {xyz} | Agent = {self.agent.vehicle.transform} | Pixel = {result} ")
        return result
        # xyz1 = np.append(xyz, np.ones(shape=(len(xyz), 1)), axis=1)
        # veh_cam_matrix = self.agent.front_depth_camera.transform.get_matrix()  # 4 x 4
        # world_veh_matrix = self.agent.vehicle.transform.get_matrix()  # 4 x 4
        # world_cam_matrix = np.linalg.inv(world_veh_matrix) @ np.linalg.inv(veh_cam_matrix)
        # cords_xyz1 = world_cam_matrix @ xyz1.T
        # raw_p2d = np.linalg.inv(self.agent.front_depth_camera.intrinsics_matrix) @ cords_xyz1[:3]
        # cam_cords = np.array(
        #     [raw_p2d[0, :] / raw_p2d[2, :], raw_p2d[1, :] / raw_p2d[2, :], raw_p2d[2, :]]
        # ).T
        # result = np.round(cam_cords, 0).astype(np.int64)
        # return result

    def show_first_person_visualization(self,
                                        show_num_waypoints: int = 0,
                                        show_point_cloud_ground: bool = False,
                                        ground_points: Optional[np.ndarray] = None):
        """
        Visualizes image from Front RGB Camera.
        Args:
            show_num_waypoints ():
            show_semantic_segmentation_obstacle ():
            show_semantic_segmentation_sky ():
            show_semantic_segmentation_ground ():
            show_point_cloud_ground ():
            ground_points ():

        Returns:
            None

        """
        rgb_img = self.agent.front_rgb_camera.data.copy()

        if show_point_cloud_ground and ground_points is not None:
            img_cords: np.ndarray = self.world_to_img_transform(ground_points)[:, :2]
            # ys = [342, 278, 271, 413 ,327 ,169 ,415, 747 ,507 ,311,311,311,311,311,311]
            # xs = [577, 513 ,531, 522 ,372 ,581, 470 ,484, 587, 523,524,525,526,527,528]
            # rgb_img[xs, ys] = [0, 0, 0]
            rgb_img[img_cords[:, 1], img_cords[:, 0]] = [0, 0, 0]  # TODO this aint working lol
        if self.agent.local_planner is not None and \
                0 < show_num_waypoints < len(self.agent.local_planner.way_points_queue):
            img_positions = self.world_to_img_transform(np.array(
                [self.agent.local_planner.way_points_queue[i].location.to_array() for i in range(show_num_waypoints)]))
            for y, x, _ in img_positions:
                rgb_img[x - 2: x + 2, y - 2: y + 2] = self.GREEN

        cv2.imshow("First Person View", rgb_img)
        cv2.waitKey(1)

    def show_birds_eye_visualization(self, focus_on_vehicle: bool = True, view_size: int = 200):
        """
        Visualizes top down image of Agent.
        Args:
            focus_on_vehicle ():
            view_size ():

        Returns:
            None
        """
        if self.occupancy_grid_map is None:
            self.logger.error("No Occupancy Grid Map is connected")
        else:
            if focus_on_vehicle:
                occu_cord = self.occupancy_grid_map.location_to_occu_cord(
                    location=self.agent.vehicle.transform.location)
                map_copy = self.occupancy_grid_map._map.copy()
                x, y = occu_cord[0]
                map_copy[
                y - self.occupancy_grid_map._vehicle_height // 2: y + self.occupancy_grid_map._vehicle_height // 2,
                x - self.occupancy_grid_map._vehicle_width // 2:x + self.occupancy_grid_map._vehicle_width // 2] = 0
                cv2.imshow("Occupancy Grid Map", map_copy[
                                                 y - view_size // 2: y + view_size // 2:,
                                                 x - view_size // 2: x + view_size // 2
                                                 ])

            else:
                cv2.imshow("Occupancy Grid Map", self.occupancy_grid_map._map)
            cv2.waitKey(1)
