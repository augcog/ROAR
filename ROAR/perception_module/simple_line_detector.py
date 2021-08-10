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
                img = orig[:, :, :]
                cv2.imshow("roi", img)
                cv2.waitKey(1)
                # b g r
                mask = cv2.inRange(img, (0, 200, 200), (120, 255, 255))
                self.agent.kwargs["lane_mask"] = mask
                return mask
            except Exception as e:
                self.logger.info(f"Unable to produce lane mask {e}")
        return None