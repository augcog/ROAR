from typing import Tuple, Any, Union

from ROAR.planning_module.abstract_planner import AbstractPlanner
import numpy as np
import cv2


class RGBPixelPlanner(AbstractPlanner):

    def run_in_series(self, **kwargs):
        if "lane_mask" in kwargs:
            img = kwargs["lane_mask"]
            cv2.imshow("mask", img)
            cv2.waitKey(1)
            mid_ys = [i for i in range(0, img.shape[0], 10)]
            straight_x = img.shape[1] // 2

            errors = []
            for y in mid_ys:
                e = self.find_error(y, img, straight_x)
                errors.append(e)
            total_error = sum(errors)
            self.agent.kwargs["lat_error"] = total_error
            return total_error

    def find_error(self, y, image, straight_x):
        # if im right on top of the line
        if image[y][straight_x] > 255:
            return 0
        # search from center to left
        for x in range(straight_x, 0, -20):
            if image[y][x] > 10:
                return x - straight_x
        # search from center to right
        for x in range(straight_x, image.shape[1], 20):
            if image[y][x] > 10:
                return x-straight_x

        # if i cant find ANY
            # if i was turning left
        if self.agent.vehicle.control.steering < -0.1:
            return 0 - straight_x
        elif self.agent.vehicle.control.steering > 0.1:
            return image.shape[1] - straight_x
        return 0

