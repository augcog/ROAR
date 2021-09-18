from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np
from ROAR.control_module.simple_pid_controller import SimplePIDController
from ROAR.perception_module.simple_line_detector import LineDetector
from ROAR.planning_module.local_planner.rgb_pixel_planner import RGBPixelPlanner
from collections import deque
from typing import List, Tuple, Optional


class LineFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        # declare color tolerance
        # BGR
        # self.lower_range = (0, 0, 170)  # low range of color
        # self.upper_range = (130, 130, 255)  # high range of color
        self.lower_range = (0, 160, 160)  # low range of color
        self.upper_range = (120, 255, 255)  # high range of color
        self.controller = SimplePIDController(agent=self)
        self.prev_steerings: deque = deque(maxlen=10)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:
            # make rgb and depth into the same shape
            depth_data = self.front_depth_camera.data
            rgb_data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))
            # find the lane
            error_at_10 = self.find_error_at(data=rgb_data,
                                             y_offset=10,
                                             lower_range=self.lower_range,
                                             upper_range=self.upper_range,
                                             error_scaling=[
                                                 (20, 0.1),
                                                 (40, 0.75),
                                                 (60, 1),
                                                 (80, 1.25),
                                                 (100, 1.5),
                                                 (200, 3)
                                             ])
            error_at_50 = self.find_error_at(data=rgb_data,
                                             y_offset=50,
                                             lower_range=self.lower_range,
                                             upper_range=self.upper_range,
                                             error_scaling=[
                                                 (20, 0.1),
                                                 (40, 0.2),
                                                 (60, 0.6),
                                                 (70, 0.7),
                                                 (80, 0.8),
                                                 (100, 1),
                                                 (200, 3)
                                             ]
            )

            if error_at_10 is None and error_at_50 is None:
                # did not see the line
                if self.vehicle.transform.rotation.pitch > -8:
                    # is flat or up slope, execute adjusted previous command
                    return self.execute_prev_command()
                else:
                    # is down slope, execute previous command as-is
                    return self.vehicle.control

            # we only want to follow the furthest thing we see.
            error = 0
            if error_at_10 is not None:
                error = error_at_10
            if error_at_50 is not None:
                error = error_at_50
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
        mask = cv2.inRange(src=data, lowerb=lower_range, upperb=upper_range)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
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
                # print(f"Error at {y_offset} -> {error, scale}")
                error = error * scale
                break
        return error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        self.logger.info("Cannot see line, executing prev cmd")
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = 0.2
        # self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control
