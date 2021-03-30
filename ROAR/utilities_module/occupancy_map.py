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
from scipy.sparse.dok import dok_matrix
from ROAR.utilities_module.module import Module
import time
from datetime import datetime
from pathlib import Path


class OccupancyGridMap(Module):
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

    def __init__(self, absolute_maximum_map_size=10000, map_padding: int = 40, vehicle_width=5, vehicle_height=5,
                 world_coord_resolution=1, occu_prob: float = 0.95, free_prob: float = 0.05,
                 max_points_to_convert: int = 1000, **kwargs):
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
        super().__init__(name="occupancy_map", **kwargs)
        self.logger = logging.getLogger(__name__)
        self._map: Optional[np.ndarray] = None
        self._world_coord_resolution = world_coord_resolution
        self._absolute_maximum_map_size = absolute_maximum_map_size

        self._min_x = -math.floor(self._absolute_maximum_map_size)
        self._min_y = -math.floor(self._absolute_maximum_map_size)

        self._max_x = math.ceil(self._absolute_maximum_map_size)
        self._max_y = math.ceil(self._absolute_maximum_map_size)

        self._map_additiona_padding = map_padding

        self._vehicle_width = vehicle_width
        self._vehicle_height = vehicle_height

        self._initialize_map()
        self._occu_prob = np.log(occu_prob / (1 - occu_prob))
        self._free_prob = 1 - self._occu_prob

        self._max_points_to_convert = max_points_to_convert
        self.curr_obstacle_world_coords = None
        self._curr_obstacle_occu_coords = None

    def _initialize_map(self):
        x_total = self._max_x - self._min_x + 2 * self._map_additiona_padding
        y_total = self._max_y - self._min_y + 2 * self._map_additiona_padding
        self._map = np.zeros(shape=(x_total, y_total),
                             dtype=np.half)  # dok_matrix((x_total, y_total), dtype=np.float32)
        self.logger.debug(f"Occupancy Grid Map of size {x_total} x {y_total} "
                          f"initialized")

    def location_to_occu_cord(self, location: Location):
        return self.cord_translation_from_world(world_cords_xy=
                                                np.array([[location.x,
                                                           location.z]]) * self._world_coord_resolution)

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
        self.logger.debug(f"Updating Grid Map: {np.shape(world_cords_xy)}")
        self._curr_obstacle_occu_coords = self.cord_translation_from_world(
            world_cords_xy=world_cords_xy)
        occu_cords_x, occu_cords_y = self._curr_obstacle_occu_coords[:, 0], self._curr_obstacle_occu_coords[:, 1]
        self._map[occu_cords_y, occu_cords_x] = 1
        # activate the below three line in real world due to sensor error
        # min_occu_cords_x, max_occu_cords_x = np.min(occu_cords_x), np.max(occu_cords_x)
        # min_occu_cords_y, max_occu_cords_y = np.min(occu_cords_y), np.max(occu_cords_y)
        # self.map[min_occu_cords_y: max_occu_cords_y, min_occu_cords_x:max_occu_cords_x] += self.free_prob
        # self.map[occu_cords_y, occu_cords_x] += self.occu_prob
        # self.map = self.map.clip(0, 1)

    def visualize(self,
                  vehicle_location: Optional[Location] = None,
                  view_size=200):
        if vehicle_location is None:
            cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(self._map), dsize=(500, 500)))
        else:
            occu_cord = self.location_to_occu_cord(
                location=vehicle_location)
            map_copy = self._map.copy()
            x, y = occu_cord[0]
            map_copy[
            y - math.floor(self._vehicle_height / 2): y + math.ceil(self._vehicle_height / 2),
            x - math.floor(self._vehicle_width / 2):x + math.ceil(self._vehicle_width / 2)] = 1
            # cv2.imshow("Occupancy Grid Map", map_copy[
            #                                  y - view_size // 2: y + view_size // 2:,
            #                                  x - view_size // 2: x + view_size // 2
            #                                  ])
            cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(map_copy), (500, 500)))
        cv2.waitKey(1)

    def update(self, world_coords: np.ndarray):
        """
        This is an easier to use update_grid_map method that can be directly called by an agent
        It will update grid map using the update_grid_map_from_world_cord method
        Args:
            world_coords: N x 3 array of points
        Returns:
            None
        """
        indices_to_select = np.random.choice(np.shape(world_coords)[0], size=min(self._max_points_to_convert,
                                                                                 np.shape(world_coords)[0]))
        world_coords = world_coords[indices_to_select]
        world_coords_xy = world_coords[:, [0, 2]] * self._world_coord_resolution
        self._update_grid_map_from_world_cord(world_cords_xy=world_coords_xy)

    def record(self, map_xs, map_ys):
        m: np.ndarray = np.zeros(shape=np.shape(self._map))
        m[map_ys, map_xs] = 1

    def run_in_series(self, **kwargs):
        if self.curr_obstacle_world_coords is not None:
            self.update(world_coords=self.curr_obstacle_world_coords)

    def update_async(self, world_coords: np.ndarray):
        """
        This is an easier to use update_grid_map method that can be directly called by an agent
        It will update grid map using the update_grid_map_from_world_cord method
        Args:
            world_coords: N x 3 array of points
        Returns:
            None
        """
        self.curr_obstacle_world_coords = world_coords

    def save(self, **kwargs):
        if self._curr_obstacle_occu_coords is not None:
            m = np.zeros(shape=self._map.shape)
            occu_cords_x, occu_cords_y = self._curr_obstacle_occu_coords[:, 0], self._curr_obstacle_occu_coords[:, 1]
            m[occu_cords_y, occu_cords_x] = 1
            np.save(f"{self.saving_dir_path}/{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}", m)

            meta_data_fpath = Path(f"{self.saving_dir_path}/meta_data.npy")

            if meta_data_fpath.exists() is False:
                meta_data = np.array([self._min_x, self._min_y, self._max_x, self._max_y, self._map_additiona_padding])
                np.save(meta_data_fpath.as_posix(), meta_data)
