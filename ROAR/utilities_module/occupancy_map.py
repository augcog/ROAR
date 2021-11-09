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
from scipy import sparse
from scipy.ndimage import rotate
from collections import deque
import itertools
import time
from ROAR.agent_module.agent import Agent
from pydantic import BaseModel, Field
import json
from PIL import Image


class OccupancyGridMap(Module):
    def __init__(self, agent: Agent, **kwargs):
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
        self._agent = agent
        config = OccupancyGridMapConfig.parse_file(self._agent.agent_settings.occu_map_config_path)
        self._map: Optional[np.ndarray] = None
        self._world_coord_resolution = config.world_coord_resolution
        self._absolute_maximum_map_size = config.absolute_maximum_map_size

        self._min_x = -math.floor(self._absolute_maximum_map_size)
        self._min_y = -math.floor(self._absolute_maximum_map_size)

        self._max_x = math.ceil(self._absolute_maximum_map_size)
        self._max_y = math.ceil(self._absolute_maximum_map_size)

        self._map_additiona_padding = config.map_padding

        self._vehicle_width = config.vehicle_width
        self._vehicle_height = config.vehicle_height

        self._initialize_map()
        self._occu_prob = np.clip(np.log(config.occu_prob / (1 - config.occu_prob)), 0, 1)
        self._free_prob = - (1 - self._occu_prob)

        self._max_points_to_convert = config.max_points_to_convert
        self.curr_obstacle_world_coords = None
        self._curr_obstacle_occu_coords = None
        self._static_obstacles: Optional[np.ndarray] = None

    def _initialize_map(self):
        x_total = self._max_x - self._min_x + 2 * self._map_additiona_padding
        y_total = self._max_y - self._min_y + 2 * self._map_additiona_padding
        self._map = np.zeros(shape=(x_total, y_total),
                             dtype=np.half)  # dok_matrix((x_total, y_total), dtype=np.float32)
        # self.logger.debug(f"Occupancy Grid Map of size {x_total} x {y_total} "
        #                   f"initialized")

    def location_to_occu_cord(self, location: Location):
        return self.cord_translation_from_world(world_cords_xy=
                                                np.array([[location.x,
                                                           location.z]]) * self._world_coord_resolution)

    def locations_to_occu_cord(self, locations: List[Location]):
        return self.cord_translation_from_world(world_cords_xy=
                                                np.array([[location.x, location.z]
                                                          for location in locations]) *
                                                self._world_coord_resolution)

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
        try:
            # self.logger.debug(f"Updating Grid Map: {np.shape(world_cords_xy)}")
            # print(f"Updating Grid Map: {np.shape(world_cords_xy)}")
            # print(len(self._curr_obstacle_occu_coords))
            if world_cords_xy is not None and len(world_cords_xy) > 0:
                self._curr_obstacle_occu_coords = self.cord_translation_from_world(
                    world_cords_xy=world_cords_xy)

                occu_cords_x, occu_cords_y = self._curr_obstacle_occu_coords[:, 0], self._curr_obstacle_occu_coords[:,
                                                                                    1]
                min_x, max_x, min_y, max_y = np.min(occu_cords_x), np.max(occu_cords_x), \
                                             np.min(occu_cords_y), np.max(occu_cords_y)
                self._map[min_y:max_y, min_x:max_x] = 0  # free
                self._map[occu_cords_y, occu_cords_x] += self._occu_prob
                # self._map = self._map.clip(0, 1)
        except Exception as e:
            self.logger.error(f"Unable to update: {e}")

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
            sA = sparse.csr_matrix(m)
            # np.save(f"{self.saving_dir_path}/{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}", m)
            sparse.save_npz(f"{self.saving_dir_path}/frame_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}", sA)
            meta_data_fpath = Path(f"{self.saving_dir_path}/meta_data.npy")

            if meta_data_fpath.exists() is False:
                meta_data = np.array([self._min_x, self._min_y, self._max_x, self._max_y, self._map_additiona_padding])
                np.save(meta_data_fpath.as_posix(), meta_data)

    def visualize(self,
                  transform: Optional[Transform] = None,
                  view_size: Tuple[int, int] = (10, 10),
                  vehicle_value: Optional[int] = None):
        """
        if transform is None:
            Visualize the entire map, output_oct_10 size constraint to (500,500)
        else:
            Visualize an ego centric view, output_oct_10 size constraint to (500,500)
        Args:
            transform: vehicle transform
            view_size: size of the view

        Returns:

        """
        curr_map = self.get_map(transform=transform, view_size=view_size, vehicle_value=vehicle_value)
        try:
            cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(curr_map), dsize=(500, 500)))
            cv2.waitKey(1)
        except Exception as e:
            print(np.shape(curr_map))
            print(e)

    def get_map(self,
                transform: Optional[Transform] = None,
                view_size: Tuple[int, int] = (100, 100),
                boundary_size: Tuple[int, int] = (100, 100),
                vehicle_value: Optional[int] = None,
                arbitrary_locations: Optional[List[Location]] = None,
                arbitrary_point_value: Optional[float] = None) -> np.ndarray:
        """
        Return global occu map if transform is None
        Otherwise, return ego centric map

        Args:
            arbitrary_point_value:
            arbitrary_locations:
            vehicle_value:
            boundary_size:
            transform: Current vehicle Transform
            view_size: Size of the view window

        Returns:
            np.ndarray of float32
        """
        if transform is None:
            return np.float32(self._map.copy())
        else:
            map_to_view = np.float32(self._map.copy())
            occu_cord = self.location_to_occu_cord(
                location=transform.location)
            x, y = occu_cord[0]
            if vehicle_value is not None:
                map_to_view[y, x] = vehicle_value

            if arbitrary_point_value is not None and arbitrary_locations is not None:
                for location in arbitrary_locations:
                    coord = self.location_to_occu_cord(location=location)[0]
                    map_to_view[coord[1], coord[0]] = arbitrary_point_value

            first_cut_size = (view_size[0] + boundary_size[0], view_size[1] + boundary_size[1])
            map_to_view = map_to_view[y - first_cut_size[1] // 2: y + first_cut_size[1] // 2,
                          x - first_cut_size[0] // 2: x + first_cut_size[0] // 2]
            image = Image.fromarray(map_to_view)
            image = image.rotate(-transform.rotation.yaw)
            map_to_view = np.asarray(image)
            # # map_to_view = np.rint(map_to_view)
            x_extra, y_extra = boundary_size[0] // 2, boundary_size[1] // 2
            map_to_view = map_to_view[y_extra: map_to_view.shape[1] - y_extra,
                          x_extra: map_to_view.shape[0] - x_extra]
            return map_to_view

    def cropped_occu_to_world(self,
                              cropped_occu_coord: np.ndarray,
                              vehicle_transform: Transform,
                              occu_vehicle_center: np.ndarray):

        diff = cropped_occu_coord - occu_vehicle_center
        vehicle_occu_coord = self.location_to_occu_cord(
            location=vehicle_transform.location)
        coord = np.array(vehicle_occu_coord[0]) + diff
        coord = coord + [self._min_x, self._min_y]
        return Transform(location=Location(x=coord[0], y=0, z=coord[1]))
        # return self.occu_to_world(occu_coord=np.array([coord[1], coord[0]]), transform=vehicle_transform)

    def load_from_file(self, file_path: Path):
        """
        Load a map from file_path.

        Expected to be the same size as the map

        Args:
            file_path: a npy file that stores the static map

        Returns:

        """
        m = np.load(file_path.as_posix())
        assert m.shape == self._map.shape, f"Loaded map is of shape [{m.shape}], " \
                                           f"does not match the expected shape [{self._map.shape}]"
        self._map = m
        self._static_obstacles = np.vstack([np.where(self._map == 1)]).T


class OccupancyGridMapConfig(BaseModel):
    absolute_maximum_map_size: int = Field(default=10000)
    map_padding: int = Field(default=40)
    vehicle_height: int = Field(default=2)
    vehicle_width: int = Field(default=2)
    world_coord_resolution: int = Field(default=1)
    occu_prob: float = Field(default=0.7)
    max_points_to_convert: int = Field(default=1000)
    update_interval: float = Field(default=0.1)
