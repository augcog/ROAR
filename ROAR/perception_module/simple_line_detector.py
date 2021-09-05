from typing import Any
import matplotlib.pyplot as plt
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent


class LineDetector(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)

    def run_in_series(self, **kwargs):
        if self.agent.front_rgb_camera.data is not None:
            try:
                orig = self.agent.front_rgb_camera.data.copy()
                shape = orig.shape
                img = orig[0:shape[0], 0:shape[1], :]  # you can limit your field of view here
                # b g r
                low_range = (0, 180, 200)  # low range of color
                high_range = (120, 255, 255)  # high range of color
                mask = cv2.inRange(img, low_range, high_range)
                self.agent.kwargs["lane_mask"] = mask

                return mask
            except Exception as e:
                self.logger.info(f"Unable to produce lane mask {e}")
        return None
