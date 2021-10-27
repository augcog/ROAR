from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
from typing import Tuple, Optional


class TrafficLightDectectorAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.min_red_radius = 70
        self.max_red_radius = 100
        self.min_green_radius = 70
        self.max_green_radius = 100
        self.should_show = True

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        if self.front_rgb_camera.data is not None:
            frame = self.front_rgb_camera.data.copy()
            red_circles, green_circles = self.detectHSV(frame)
            has_red_circles, has_green_circles = red_circles is not None, green_circles is not None
            print(f"has_red_light = {has_red_circles} | has_green_light = {has_green_circles}")
            return VehicleControl(throttle=0, steering=0)

        else:
            return VehicleControl(throttle=0.5, steering=0.5)

    def detectHSV(self, img: np.ndarray) -> Tuple[Optional[list], Optional[list]]:
        cimg = img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 90, 90])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])

        # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskr = mask2  # cv2.add(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        maskr = cv2.erode(maskr, kernel, iterations=1)
        maskg = cv2.inRange(hsv, lower_green, upper_green)

        # hough circle detect
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80, param1=80, param2=10,
                                     minRadius=self.min_red_radius, maxRadius=self.max_red_radius)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=self.min_green_radius,
                                     maxRadius=self.max_green_radius)
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))
            for i in r_circles[0, :]:
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))
            for i in g_circles[0, :]:
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)
        if self.should_show:
            cv2.imshow('detected', cimg)
            # cv2.imshow('maskr', maskr)

        return r_circles, g_circles
