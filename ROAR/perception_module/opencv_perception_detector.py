from typing import Any
from ROAR.agent_module.agent import Agent
import cv2
from ROAR.perception_module.detector import Detector


class OpenCVObjectDetector(Detector):
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)

    def run_in_series(self, **kwargs) -> Any:
        front_rgb = self.agent.front_rgb_camera.data
        if front_rgb is not None:
            cv2.imshow("im", front_rgb)
            cv2.waitKey(1)
