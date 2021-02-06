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
    Log update Occupancy map
    """

    def __init__(self, scale: float = 1.0, buffer_size: int = 10, occu_prob: float = 0.65, free_prob: float = 0.35):
        self.scale = scale
        self.curr_min_world_coord: float = 0
        self.curr_max_world_coord: float = 100
        self.buffer_size = buffer_size
        self.occu_prob = np.log(occu_prob / (1 - occu_prob))
        self.free_prob = np.log(free_prob / (1 - free_prob))
        self._map = self._create_empty_map(shape=
                                           (math.ceil(self.curr_max_world_coord - self.curr_min_world_coord),
                                            math.ceil(self.curr_max_world_coord - self.curr_min_world_coord)))
        self.logger = logging.getLogger("Occupancy Grid Map")

    def _create_empty_map(self, shape: Tuple[int, int]):
        return np.zeros(shape=(shape[0] + self.buffer_size, shape[1] + self.buffer_size))

    def update(self, world_coords: np.ndarray) -> bool:
        try:
            # print(np.amin(world_coords, axis=0), np.amax(world_coords, axis=0))
            scaled_world_coords = world_coords * self.scale

            # rescale world coord to min = 0
            Xs = scaled_world_coords[:, 0]
            Zs = scaled_world_coords[:, 1]

            # rescale occupancy map if neccessary
            min_coord = min(np.min(Xs), np.min(Zs))
            max_coord = max(np.max(Xs), np.max(Zs))

            translated_Xs = np.array(Xs + abs(min_coord), dtype=np.int)
            translated_Zs = np.array(Zs + abs(min_coord), dtype=np.int)

            if min_coord < self.curr_min_world_coord or max_coord > self.curr_max_world_coord:
                self._rescale_occupancy_map(min_coord=min_coord, max_coord=max_coord)

            # plot transformed world coord
            occupied_mask = self._map.copy() # np.zeros(shape=self._map.shape) # self._map.copy()
            occupied_mask[translated_Xs, translated_Zs] = 1
            cv2.imshow("occupied mask", cv2.resize(occupied_mask, dsize=(500,500)))
            cv2.waitKey(1)
            # print(np.amin(translated_Xs), np.amax(translated_Xs), np.amin(translated_Zs), np.amax(translated_Zs),
            #       self._map.shape)
            self._map[occupied_mask == 1] += self.occu_prob
            self._map[occupied_mask == 0] += self.free_prob
            self._map.clip(min=0, max=1)
            return True
        except Exception as e:
            self.logger.error(f"Something went wrong: {e}")
            return False

    def _rescale_occupancy_map(self, min_coord: float, max_coord: float, auto_scale_factor=1):
        auto_scale_factor = 1 if auto_scale_factor < 1 else auto_scale_factor
        min_coord = min(self.curr_min_world_coord, min_coord)
        max_coord = max(self.curr_max_world_coord, max_coord)

        new_map_shape = (math.ceil((max_coord - min_coord) * auto_scale_factor),
                         math.ceil((max_coord - min_coord) * auto_scale_factor))

        # copy current occupancy map
        curr_occu_map_copy: np.ndarray = self._map.copy()

        # create new map that includes min and max coord
        new_map: np.ndarray = self._create_empty_map(shape=new_map_shape)
        if new_map.shape == self._map.shape:
            return
        offset = math.ceil(self.curr_min_world_coord - min_coord)

        new_map[offset:offset + curr_occu_map_copy.shape[0], offset:offset + curr_occu_map_copy.shape[1]] = \
            curr_occu_map_copy

        # reset curr_min_world_coord and curr_max_world_coord
        self.curr_min_world_coord = min_coord
        self.curr_max_world_coord = max_coord

        # reset current_occupancy map
        self._map = new_map

    def vizualize(self, center: Tuple[float, float], view_width: int = 20):
        occu_map_center_x = min(0, int(center[0] + abs(self.curr_min_world_coord)))
        occu_map_center_y = min(0, int(center[1] + abs(self.curr_min_world_coord)))

        min_x, min_y, max_x, max_y = min(0, occu_map_center_x - view_width), \
                                     min(0, occu_map_center_y - view_width), \
                                     max(self._map.shape[0], occu_map_center_x + view_width), \
                                     max(self._map.shape[1], occu_map_center_y + view_width)
        # image = self._map[min_x:max_x, min_y:max_y]
        cv2.imshow("occupancy map", cv2.resize(self._map, dsize=(500,500)))
        cv2.waitKey(1)
