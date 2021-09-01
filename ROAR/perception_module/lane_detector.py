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
        # red
        self.lower_range = (20, 20, 190)  # low range of color
        self.upper_range = (120, 120, 255)  # high range of color

        # yellow
        # self.lower_range = (0, 200, 200)  # low range of color
        # self.upper_range = (150, 240, 255)  # high range of color

    def run_in_series(self, **kwargs) -> Optional[np.ndarray]:
        if self.agent.front_depth_camera.data is not None and self.agent.front_rgb_camera.data is not None:
            depth_data = self.agent.front_depth_camera.data
            rgb_data: np.ndarray = cv2.resize(self.agent.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))

            y = rgb_data.shape[0] - 10
            Xs = [x for x in range(0, rgb_data.shape[1], 5)]
            i = 0
            while i < len(Xs):
                x = Xs[i]
                if self.lower_range[0] < rgb_data[y][x][0] < self.upper_range[0] \
                        and self.lower_range[1] < rgb_data[y][x][1] < self.upper_range[1] \
                        and self.lower_range[2] < rgb_data[y][x][2] < self.upper_range[2]:
                    i += 1  # move it in more so that it is right on the line
                    break
                i += 1
            if i >= len(Xs):
                # no correct pixel found
                return None

            # rgb_data = cv2.circle(rgb_data, center=(Xs[i],y), radius=2, color=(0,0,0), thickness=-1)
            # cv2.imshow("rgb_data", rgb_data)

            h, w, chn = rgb_data.shape
            seed = (Xs[i], y)
            mask = np.zeros(shape=(h + 2, w + 2), dtype=np.uint8)
            floodflags = 8
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
