from typing import Any
import matplotlib.pyplot as plt
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent
from typing import Optional

class LaneDetector(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.lower_range = (20, 20, 190)  # low range of color
        self.upper_range = (120, 120, 255)  # high range of color

    def run_in_series(self, **kwargs) -> Optional[np.ndarray]:
        if self.agent.front_depth_camera.data is not None and self.agent.front_rgb_camera.data is not None:

            depth_data = self.agent.front_depth_camera.data
            rgb_data: np.ndarray = cv2.resize(self.agent.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))
            # find strip loc around base
            img = rgb_data
            mask = cv2.inRange(src=img, lowerb=self.lower_range , upperb=self.upper_range)

            locs = np.where(mask > 0.5)
            y_index = np.argmin(locs[1])
            # img = cv2.circle(rgb_data, center=seed, radius=10, color=(0,0,0), thickness=-1)

            h, w, chn = rgb_data.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            seed = (locs[1][y_index] + 2, locs[0][y_index])
            floodflags = 4
            floodflags |= cv2.FLOODFILL_MASK_ONLY
            floodflags |= (255 << 8)
            num, im, mask, rect = cv2.floodFill(image=rgb_data,
                                                mask=mask,
                                                seedPoint=seed,
                                                newVal=(0, 0, 255),
                                                loDiff=(10,) * 3,
                                                upDiff=(10,) * 3,
                                                flags=floodflags)
            mask = mask[1:-1, 1:-1]
            return mask
        return None

