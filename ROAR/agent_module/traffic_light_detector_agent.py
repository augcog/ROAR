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
            mask_r = self.detectRed(frame)
            cv2.imshow("mask_r", mask_r)
            cv2.waitKey(1)
            num_spots_red = len(np.where(mask_r > 0.5)[0])
            has_red_circles = num_spots_red > 9000
            print(num_spots_red)

            if has_red_circles:
                return VehicleControl(brake=True)

            return VehicleControl(throttle=0.16, steering=0)

        else:
            return VehicleControl(throttle=0.16, steering=0)

    def detectRed(self, img) -> np.ndarray:
        mask_r = cv2.inRange(img, (0, 0, 140), (60, 60, 255))
        # kernel = np.ones((10, 10), np.uint8)
        # mask_r = cv2.erode(mask_r, kernel, iterations=2)
        return mask_r

    def detectHSV(self, img: np.ndarray) -> Tuple[Optional[list], Optional[list]]:
        cimg = img
        mask_r = cv2.inRange(img, (0, 0, 170), (130, 130, 255))
        kernel = np.ones((10, 10), np.uint8)
        mask_r = cv2.erode(mask_r, kernel, iterations=2)
        cv2.imshow("mask_r", mask_r)
        cv2.waitKey(1)
        r_circles = cv2.HoughCircles(mask_r, cv2.HOUGH_GRADIENT, 1, 80, param1=80, param2=10,
                                     minRadius=self.min_red_radius, maxRadius=self.max_red_radius)
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))
            for i in r_circles[0, :]:
                # draw the outer circle
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)

        return r_circles, None
