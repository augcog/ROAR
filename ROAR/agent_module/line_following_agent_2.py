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
        self.lower_range = (0, 170, 200)  # low range of color
        self.upper_range = (150, 255, 255)  # high range of color
        self.controller = SimplePIDController(agent=self)
        # self.error_scaling: List[Tuple[float, float]] = [
        #     (20, 0.1),
        #     (40, 0.75),
        #     (60, 1),
        #     (80, 1.5),
        #     (100, 1.75),
        #     (200, 3)
        # ]
        self.prev_steerings: deque = deque(maxlen=10)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:

            # make rgb and depth into the same shape
            depth_data = self.front_depth_camera.data
            rgb_data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))
            # find the lane
            error_at_10 = self.find_error_at(rgb_data=rgb_data, y_offset=10, error_scaling=[
                (20, 0.1),
                (40, 0.75),
                (60, 1),
                (80, 1.5),
                (100, 1.75),
                (200, 3)
            ])
            error_at_50 = self.find_error_at(rgb_data=rgb_data, y_offset=50, error_scaling=[
                (20, 0.1),  # TUNE THIS!
                (40, 0.75),
                (60, 1),
                (80, 1.5),
                (100, 1.75),
                (200, 3)
            ])
            if error_at_10 is None and error_at_50 is None:
                return self.execute_prev_command()

            # we only want to follow the furthest thing we see.
            error = 0
            if error_at_10 is not None:
                error = error_at_10
            if error_at_50 is not None:
                error = error_at_50

            self.kwargs["lat_error"] = error
            self.vehicle.control = self.controller.run_in_series()
            self.prev_steerings.append(self.vehicle.control.steering)
            return self.vehicle.control
        else:
            # image feed is not available yet
            return VehicleControl()

    def find_error_at(self, rgb_data, y_offset, error_scaling) -> Optional[float]:
        y = rgb_data.shape[0] - y_offset
        lane_x = []
        mask = cv2.inRange(src=rgb_data, lowerb=self.lower_range, upperb=self.upper_range)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        for x in range(0, rgb_data.shape[1], 5):
            if mask[y][x] > 0:
                lane_x.append(x)

        if len(lane_x) == 0:
            return None

        # if lane is found
        avg_x = int(np.average(lane_x))

        # find error
        center_x = rgb_data.shape[1] // 2

        error = avg_x - center_x
        # we want small error to be almost ignored, only big errors matter.
        for e, scale in error_scaling:
            if abs(error) <= e:
                error = error * scale
                break
        return error

    def execute_prev_command(self):
        # no lane found, execute the previous control with a decaying factor
        if np.average(self.prev_steerings) < 0:
            self.vehicle.control.steering = -1
        else:
            self.vehicle.control.steering = 1
        self.prev_steerings.append(self.vehicle.control.steering)
        self.vehicle.control.throttle = 0.2
        # self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
        return self.vehicle.control
