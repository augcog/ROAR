from typing import Tuple, Any, Union

from ROAR.planning_module.abstract_planner import AbstractPlanner
import numpy as np
import cv2


class RGBPixelPlanner(AbstractPlanner):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.visualize = True

    def run_in_series(self, **kwargs):
        if "lane_mask" in kwargs:
            lane_mask = kwargs["lane_mask"]
            s = np.shape(lane_mask)
            img = lane_mask[40*s[0]//100:100*s[0]//100, 0:s[1]]
            empty = np.ones(shape=np.shape(img))
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)  # find the largest contour
                x, y, w, h = cv2.boundingRect(c)
                x_avg = x + (w // 2)
                if self.visualize:
                    cv2.drawContours(empty, contours, -1, 0, 3)  # draw out the rectangle
                    cv2.imshow("lane_mask", lane_mask)
                    cv2.rectangle(empty, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    cv2.circle(empty, (x_avg, y), radius=10, thickness=-1, color=0)
                    cv2.imshow("next_waypoint", empty)
                    cv2.waitKey(1)

                straight_x = img.shape[1] // 2

                return x_avg - straight_x
            return 0

    def find_error(self, y, image, straight_x):
        # if im right on top of the line
        if image[y][straight_x] > 200:
            return 0
        # search from center to left
        for x in range(straight_x, 0, -20):
            if image[y][x] > 10:
                return x - straight_x
        # search from center to right
        for x in range(straight_x, image.shape[1], 20):
            if image[y][x] > 10:
                return x - straight_x

        # if i cant find ANY
        # if i was turning left
        if self.agent.vehicle.control.steering < -0.1:
            return 0 - straight_x
        elif self.agent.vehicle.control.steering > 0.1:
            return image.shape[1] - straight_x
        return 0
