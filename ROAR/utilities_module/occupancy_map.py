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

    def __init__(self, scale: float = 1.0, buffer_size: int = 50, occu_prob: float = 0.65, free_prob: float = 0.35):
        self.scale = scale
        self.min_x: int = 0
        self.min_y: int = 0
        self.max_x: int = 100
        self.max_y: int = 100
        self.buffer_size = buffer_size
        self.occu_prob = np.log(occu_prob / (1 - occu_prob))
        self.free_prob = np.log(free_prob / (1 - free_prob))
        self._map: np.ndarray = self._create_empty_map(shape=(math.ceil(self.max_x - self.min_x),
                                                              math.ceil(self.max_y - self.min_y)))
        self.logger = logging.getLogger("Occupancy Grid Map")

    @staticmethod
    def _create_empty_map(shape: Tuple[int, int], buffer_size=10) -> np.ndarray:
        return np.zeros(shape=(shape[0] + buffer_size, shape[1] + buffer_size))

    def update(self, world_coords: np.ndarray, vehicle_location: Optional[Location] = None) -> bool:
        try:
            scaled_world_coords = world_coords * self.scale

            # rescale world coord to min = 0
            Xs = scaled_world_coords[:, 0]
            Zs = scaled_world_coords[:, 2]

            min_x, max_x = int(np.min(Xs)), int(np.max(Xs))
            min_y, max_y = int(np.min(Zs)), int(np.max(Zs))

            map_size = (max_x-min_x + self.buffer_size, max_y-min_y+self.buffer_size)
            translated_Xs = np.array(Xs - min_x, dtype=np.int)
            translated_Ys = np.array(Zs - min_y, dtype=np.int)

            vehicle_occ_x = int(vehicle_location.x - min_x)
            vehicle_occ_y = int(vehicle_location.z - min_y)

            curr_map = np.zeros(shape=map_size)
            curr_map[translated_Xs, translated_Ys] = 1
            curr_map[vehicle_occ_x-1:vehicle_occ_x+1, vehicle_occ_y-1:vehicle_occ_y+1] = 1
            self._map = curr_map
            # cv2.imshow("curr_map", cv2.resize(curr_map, (500,500)))
            # cv2.waitKey(1)

            # print("BEFORE RESCALING: ", max_x - min_x, max_y - min_y)

            # if min_x < self.min_x or min_y < self.min_y or max_x > self.max_x or max_y > self.max_y:
            #     self.rescale_map(new_min_x=min_x, new_max_x=max_x, new_min_y=min_y, new_max_y=max_y)
            #
            # # translate to occupancy map coordinate
            # translated_Xs, translated_Ys = self.to_occupancy_map_coor(Xs, Zs)
            #
            # self._map = np.zeros(shape=self._map.shape)
            # self._map[translated_Xs, translated_Ys] = 1

            # translated_Xs = np.array(Xs - min_x, dtype=np.int)
            # translated_Ys = np.array(Zs - min_y, dtype=np.int)
            # # print("AFTER RESCALING: ", np.max(translated_Xs) - np.min(translated_Xs),
            # #       np.max(translated_Ys) - np.min(translated_Ys), np.shape(self._map))
            # occupied_mask = np.zeros(shape=self._map.shape)
            # occupied_mask[translated_Xs, translated_Ys] = 1
            #
            # self._map[occupied_mask == 1] += self.occu_prob
            # self._map[occupied_mask == 0] += self.free_prob

            # self._map.clip(min=0, max=1)
            # # cv2.imshow("occupied_mask", cv2.resize(occupied_mask, (500, 500)))
            # cv2.imshow("map", cv2.resize(self._map, (500, 500)))
            # cv2.waitKey(1)

            return True
        except Exception as e:
            self.logger.error(f"Something went wrong: {e}")
            return False

    def rescale_map(self, new_min_x: int, new_max_x: int, new_min_y: int, new_max_y: int):
        old_width, old_height = self._map.shape
        old_min_x, old_max_x, old_min_y, old_max_y = self.min_x, self.max_x, self.min_y, self.max_y

        new_min_x, new_min_y = min(new_min_x, old_min_x), min(new_min_y, old_min_y)
        new_max_x, new_max_y = max(new_max_x, old_max_x), max(new_max_y, old_max_y)

        new_map_size = (math.ceil(new_max_x - new_min_x), math.ceil(new_max_y - new_min_y))
        new_map = self._create_empty_map(shape=new_map_size, buffer_size=self.buffer_size)

        new_map[(old_min_x - new_min_x):  (old_min_x - new_min_x) + old_width,
        (old_min_y - new_min_y):  (old_min_y - new_min_y) + old_height] = self._map

        self._map = new_map
        self.min_x, self.min_y, self.max_x, self.max_y = new_min_x, new_min_y, new_max_x, new_max_y

    def visualize(self):
        map_copy = self._map.copy()
        cv2.imshow("map", cv2.resize(map_copy, (500,500)))
        cv2.waitKey(1)
    def to_occupancy_map_coor(self, Xs, Ys, ) -> Tuple[np.ndarray, np.ndarray]:
        translated_Xs = np.array(Xs - self.min_x, dtype=np.int)
        translated_Ys = np.array(Ys - self.min_y, dtype=np.int)
        return translated_Xs, translated_Ys


