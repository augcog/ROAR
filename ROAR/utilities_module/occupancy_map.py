import numpy as np
from pydantic import BaseModel, Field
from typing import Union, List, Tuple
from ROAR.utilities_module.data_structures_models import Transform, Location
import cv2
import logging
import math
from typing import Optional, List
from ROAR.utilities_module.camera_models import Camera
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR.utilities_module.utilities import img_to_world
import logging
import time


class OccupancyGridMap:
    """
    A 2D Occupancy Grid Map representing the world
    Should be able to handle
        1. Transformation of coordinate from world to grid cord
        2. Transformation of cord from grid cord to world cord
        3. Represent the vehicle size and position, vehicle is represented as 0
            Note that this class does NOT remember the vehicle position,
            in order to visualize vehicle, vehicle position has to be passed in
        4. Represent the obstacles' position once its world coordinate is
           given
        5. Represent free space's position, once its world coordinate is given
        6. visualize itself, including zoom to a certain area so that not
           the entire map is visualized
        7. The range of values should be bewteen 0 - 1
            - 0 = obstacles, 1 = free space
        8 Failure tolerant, if I pass in a wrong world coordinate,
          it will prompt it, but do not fail. Similar with other functions
          in this class
        9. Fixed map size for enhanced performance
    """

    def __init__(self, absolute_maximum_map_size=10000,
                 map_padding: int = 40,
                 vehicle_width=5,
                 vehicle_height=5,
                 world_coord_resolution=1,
                 occu_prob: float = 0.95,
                 free_prob: float = 0.05):
        """
        Args:
            absolute_maximum_map_size: Absolute maximum size of the map, will be used to compute a square occupancy map
            map_padding: additional padding intended to add.
        Note: This method pad to both sides, for example, it create padding
        to the left of min_x, and to the right of max_x
        Note: map_padding is for when when visualizing, we have to take a whole
         block and just in case the route is to the edge of the map,
         it will not error out
        """
        self.logger = logging.getLogger(__name__)
        self.map: Optional[np.ndarray] = None
        self.world_coord_resolution = world_coord_resolution
        self._absolute_maximum_map_size = absolute_maximum_map_size

        self._min_x = -math.floor(self._absolute_maximum_map_size)
        self._min_y = -math.floor(self._absolute_maximum_map_size)

        self._max_x = math.ceil(self._absolute_maximum_map_size)
        self._max_y = math.ceil(self._absolute_maximum_map_size)

        self._map_additiona_padding = map_padding

        self.vehicle_width = vehicle_width
        self.vehicle_height = vehicle_height

        self._initialize_map()
        self.occu_prob = np.log(occu_prob / (1 - occu_prob))
        self.free_prob = 1 - self.occu_prob

    def _initialize_map(self):
        x_total = self._max_x - self._min_x + 2 * self._map_additiona_padding
        y_total = self._max_y - self._min_y + 2 * self._map_additiona_padding
        self.map = np.zeros([x_total, y_total])
        self.logger.debug(f"Occupancy Grid Map of size {x_total} x {y_total} "
                          f"initialized")

    def location_to_occu_cord(self, location: Location):
        return self.cord_translation_from_world(world_cords_xy=
                                                np.array([[location.x,
                                                           location.z]]) * self.world_coord_resolution)

    def cord_translation_from_world(self,
                                    world_cords_xy: np.ndarray) -> np.ndarray:
        """
        Translate from world coordinate to occupancy coordinate
        If the given world coord is less than min or greater than maximum,
        then do not execute the translation, log error message
        Args:
            world_cords_xy: Numpy array of shape (N, 2) representing
             [[x, y], [x, y], ...]
        Returns:
            occupancy grid map coordinate for this world coordinate of shape
            (N, 2)
            [
             [x, y],
             [x, y]
            ]
        """
        # reshape input into a ndarray that looks like [[X, Y], [X, Y]...]
        # self.logger.debug(f"translating world cords xy: {np.shape(world_cords_xy)}")
        transformed = np.round(world_cords_xy - [self._min_x, self._min_y]).astype(np.int64)
        return transformed

    def _update_grid_map_from_world_cord(self, world_cords_xy):
        """
        Updates the grid map based on the world coordinates passed in
        Args:
            world_cords_xy: Numpy array of shape (N, 2) representing
             [[x, y], [x, y], ...]
        Returns:
            None
        """
        # find occupancy map cords
        # self.logger.debug(f"Updating Grid Map: {np.shape(world_cords_xy)}")
        occu_cords = self.cord_translation_from_world(
            world_cords_xy=world_cords_xy)
        # self.logger.debug(f"Occupancy Grid Map Cord shape = {np.shape(occu_cords)}")
        occu_cords_x, occu_cords_y = occu_cords[:, 0], occu_cords[:, 1]
        min_occu_cords_x, max_occu_cords_x = np.min(occu_cords_x), np.max(occu_cords_x)
        min_occu_cords_y, max_occu_cords_y = np.min(occu_cords_y), np.max(occu_cords_y)
        self.map[min_occu_cords_y: max_occu_cords_y, min_occu_cords_x:max_occu_cords_x] -= 0.01
        self.map[occu_cords_y, occu_cords_x] += self.occu_prob
        self.map[min_occu_cords_y: max_occu_cords_y,
        min_occu_cords_x:max_occu_cords_x] = \
            self.map[min_occu_cords_y: max_occu_cords_y, min_occu_cords_x:max_occu_cords_x].clip(0, 1)

    def visualize(self,
                  vehicle_location: Optional[Location] = None,
                  view_size=200):
        if vehicle_location is None:
            cv2.imshow("Occupancy Grid Map", cv2.resize(self.map, dsize=(500, 500)))
        else:

            occu_cord = self.location_to_occu_cord(
                location=vehicle_location)
            map_copy = self.map.copy()
            x, y = occu_cord[0]
            map_copy[
            y - math.floor(self.vehicle_height / 2): y + math.ceil(self.vehicle_height / 2),
            x - math.floor(self.vehicle_width / 2):x + math.ceil(self.vehicle_width / 2)] = 1
            # cv2.imshow("Occupancy Grid Map", map_copy[
            #                                  y - view_size // 2: y + view_size // 2:,
            #                                  x - view_size // 2: x + view_size // 2
            #                                  ])
            cv2.imshow("Occupancy Grid Map", cv2.resize(map_copy, (500, 500)))
        cv2.waitKey(1)

    def update(self, world_coords: np.ndarray):
        """
        This is an easier to use update_grid_map method that can be directly called by an agent
        It will update grid map using the update_grid_map_from_world_cord method
        Args:
            world_coords:
        Returns:
            None
        """
        world_coords_xy = world_coords[:, [0, 2]] * self.world_coord_resolution
        self._update_grid_map_from_world_cord(world_cords_xy=world_coords_xy)

# class OccupancyGridMap:
#     """
#     Log update Occupancy map
#     """
#
#     def __init__(self, scale: float = 1.0, buffer_size: int = 50, occu_prob: float = 0.65, free_prob: float = 0.35):
#         self.scale = scale
#         self.min_x: int = -1000
#         self.min_y: int = -1000
#         self.max_x: int = 1000
#         self.max_y: int = 1000
#         self.buffer_size = buffer_size
#         self.occu_prob = np.log(occu_prob / (1 - occu_prob))
#         self.free_prob = np.log(free_prob / (1 - free_prob))
#         self._map: np.ndarray = self._create_empty_map(shape=(math.ceil(self.max_x - self.min_x),
#                                                               math.ceil(self.max_y - self.min_y)))
#         self.logger = logging.getLogger("Occupancy Grid Map")
#
#     @staticmethod
#     def _create_empty_map(shape: Tuple[int, int], buffer_size=10) -> np.ndarray:
#         return np.zeros(shape=(shape[0] + buffer_size, shape[1] + buffer_size))
#
#     def update(self, world_coords: np.ndarray, vehicle_location: Optional[Location] = None) -> bool:
#         try:
#             scaled_world_coords = world_coords * self.scale
#
#             # translate world coord into occupancy map coord
#             # self.to_occu_map_coord(scaled_world_coords)
#             # plot coord onto occupancy map
#
#
#             # rescale world coord to min = 0
#             # Xs = scaled_world_coords[:, 0]
#             # Zs = scaled_world_coords[:, 2]
#             #
#             # min_x, max_x = int(np.min(Xs)), int(np.max(Xs))
#             # min_y, max_y = int(np.min(Zs)), int(np.max(Zs))
#             #
#             # map_size = (max_x - min_x + self.buffer_size, max_y - min_y + self.buffer_size)
#             # translated_Xs = np.array(Xs - min_x, dtype=np.int)
#             # translated_Ys = np.array(Zs - min_y, dtype=np.int)
#             #
#             # vehicle_occ_x = int(vehicle_location.x - min_x)
#             # vehicle_occ_y = int(vehicle_location.z - min_y)
#             #
#             # curr_map = np.zeros(shape=map_size)
#             # curr_map[translated_Xs, translated_Ys] = 1
#             # print(np.amin(translated_Xs), np.amax(translated_Xs), np.amin(translated_Ys), np.amax(translated_Ys))
#             # curr_map[vehicle_occ_x - 1:vehicle_occ_x + 1, vehicle_occ_y - 1:vehicle_occ_y + 1] = 1
#             # self._map = curr_map
#
#             return True
#         except Exception as e:
#             self.logger.error(f"Something went wrong: {e}")
#             return False
#
#     def rescale_map(self, new_min_x: int, new_max_x: int, new_min_y: int, new_max_y: int):
#         old_width, old_height = self._map.shape
#         old_min_x, old_max_x, old_min_y, old_max_y = self.min_x, self.max_x, self.min_y, self.max_y
#
#         new_min_x, new_min_y = min(new_min_x, old_min_x), min(new_min_y, old_min_y)
#         new_max_x, new_max_y = max(new_max_x, old_max_x), max(new_max_y, old_max_y)
#
#         new_map_size = (math.ceil(new_max_x - new_min_x), math.ceil(new_max_y - new_min_y))
#         new_map = self._create_empty_map(shape=new_map_size, buffer_size=self.buffer_size)
#
#         new_map[(old_min_x - new_min_x):  (old_min_x - new_min_x) + old_width,
#         (old_min_y - new_min_y):  (old_min_y - new_min_y) + old_height] = self._map
#
#         self._map = new_map
#         self.min_x, self.min_y, self.max_x, self.max_y = new_min_x, new_min_y, new_max_x, new_max_y
#
#     def visualize(self):
#         map_copy = self._map.copy()
#         cv2.imshow("map", cv2.resize(map_copy, (500, 500)))
#         cv2.waitKey(1)
#
#     def to_occupancy_map_coor(self, Xs, Ys, ) -> Tuple[np.ndarray, np.ndarray]:
#         translated_Xs = np.array(Xs - self.min_x, dtype=np.int)
#         translated_Ys = np.array(Ys - self.min_y, dtype=np.int)
#         return translated_Xs, translated_Ys
