from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
# from ROAR.control_module.simple_pid_controller import SimplePIDController
from ROAR.control_module.real_world_image_based_pid_controller import RealWorldImageBasedPIDController as PIDController
from collections import deque
from typing import List, Tuple, Optional


class LineFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        # declare color tolerance
        # BGR
        # self.rgb_lower_range = (0, 0, 170)  # low range of color
        # self.rgb_upper_range = (130, 130, 255)  # high range of color
        self.rgb_lower_range = (0, 160, 160)  # low range of color YELLOW
        self.rgb_upper_range = (140, 255, 255)  # high range of color

        # (-128, -50; 0, 70) + 128
        # 150 - 200, 0 - 60; 150, 96
        # self.ycbcr_lower_range = (0, 220, 60)  # low range of color YELLOW
        # self.ycbcr_upper_range = (250, 240, 130)  # high range of color
        self.ycbcr_lower_range = (0, 180, 60)  # low range of color
        self.ycbcr_upper_range = (250, 240, 140)  # high range of color
        self.controller = PIDController(agent=self)
        self.prev_steerings: deque = deque(maxlen=10)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        # if self.front_depth_camera.data is not None:
        #     depth_data = self.front_depth_camera.data
        #     roi = depth_data[3*depth_data.shape[0]//4:, :]
        #     # cv2.imshow("roi", roi)
        #     # cv2.imshow("depth", depth_data)
        #     dist = np.average(roi)
        #     if dist < 0.25:
        #         return VehicleControl(throttle=0, steering=0)
        if self.front_rgb_camera.data is not None:
            # make rgb and depth into the same shape
            data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                          dsize=(192, 256))
            # cv2.imshow("rgb_mask", cv2.inRange(data, self.rgb_lower_range, self.rgb_upper_range))
            data = self.rgb2ycbcr(data)
            # cv2.imshow("ycbcr_mask", cv2.inRange(data, self.ycbcr_lower_range, self.ycbcr_upper_range))
            # find the lane
            error_at_10 = self.find_error_at(data=data,
                                             y_offset=10,
                                             lower_range=self.ycbcr_lower_range,
                                             upper_range=self.ycbcr_upper_range,
                                             error_scaling=[
                                                 (20, 0.1),
                                                 (40, 0.75),
                                                 (60, 0.8),
                                                 (80, 0.9),
                                                 (100, 0.95),
                                                 (200, 1)
                                             ])
            error_at_50 = self.find_error_at(data=data,
                                             y_offset=50,
                                             lower_range=self.ycbcr_lower_range,
                                             upper_range=self.ycbcr_upper_range,
                                             error_scaling=[
                                                 (20, 0.1),
                                                 (40, 0.4),
                                                 (60, 0.6),
                                                 (70, 0.7),
                                                 (80, 0.8),
                                                 (100, 0.9),
                                                 (200, 2)
                                             ]
                                             )

            if error_at_10 is None and error_at_50 is None:
                # did not see the line
                neutral = -90
                incline = self.vehicle.transform.rotation.pitch - neutral
                if incline < -10:
                    # is down slope, execute previous command as-is
                    # get the PID for downhill
                    long_control = self.controller.long_pid_control()
                    self.vehicle.control.throttle = long_control
                    return self.vehicle.control

                else:
                    # is flat or up slope, execute adjusted previous command
                    return self.execute_prev_command()

            # we only want to follow the furthest thing we see.
            error = 0
            if error_at_10 is not None:
                error = error_at_10
            if error_at_50 is not None:
                error = error_at_50

            # print(error_at_10, error_at_50, error)
            self.kwargs["lat_error"] = error
            self.vehicle.control = self.controller.run_in_series()
            self.prev_steerings.append(self.vehicle.control.steering)

            # self.logger.info(f"line recognized: {error}| control: {self.vehicle.control}")
            return self.vehicle.control
        else:
            # image feed is not available yet
            return VehicleControl()

    def find_error_at(self, data, y_offset, error_scaling, lower_range, upper_range) -> Optional[float]:
        y = data.shape[0] - y_offset
        lane_x = []
        mask_red = cv2.inRange(src=data, lowerb=lower_range, upperb=upper_range)
        mask_yellow = cv2.inRange(src=data, lowerb=(0, 140, 0), upperb=(250, 200, 80))
        mask = mask_red | mask_yellow

        cv2.imshow("mask", mask)
        # cv2.imshow("mask_red", mask_red)
        # cv2.imshow("mask_yellow", mask_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        for x in range(0, data.shape[1], 5):
            if mask[y][x] > 0:
                lane_x.append(x)

        if len(lane_x) == 0:
            return None

        # if lane is found
        avg_x = int(np.average(lane_x))

        # find error
        center_x = data.shape[1] // 2

        error = avg_x - center_x
        # we want small error to be almost ignored, only big errors matter.
        for e, scale in error_scaling:
            if abs(error) <= e:
                # print(f"Error at {y_offset} -> {error, scale} -> {error * scale}")
                error = error * scale
                break

        return error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        self.logger.info("Executing prev")

        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        # self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = self.vehicle.control.throttle
        self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control

    def rgb2ycbcr(self, im):
        xform = np.array([[.299, .587, .114],
                          [-.1687, -.3313, .5],
                          [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)
