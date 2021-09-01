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
from typing import List, Tuple


class LineFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        # declare color tolerance
        self.lower_range = (0, 170, 200)  # low range of color
        self.upper_range = (150, 240, 255)  # high range of color
        self.controller = SimplePIDController(agent=self)
        self.error_scaling: List[Tuple[float, float]] = [
            (20, 0.1),
            (40, 0.5),
            (60, 1),
            (80, 1.5),
            (100, 1.75),
            (200, 3)
        ]
        self.prev_control:VehicleControl = VehicleControl()

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None and self.front_rgb_camera.data is not None:

            # make rgb and depth into the same shape
            depth_data = self.front_depth_camera.data
            rgb_data: np.ndarray = cv2.resize(self.front_rgb_camera.data.copy(),
                                              dsize=(depth_data.shape[1], depth_data.shape[0]))
            # find the lane
            y = rgb_data.shape[0] - 10
            lane_x = []
            for x in range(0, rgb_data.shape[1], 5):
                if self.lower_range[0] < rgb_data[y][x][0] < self.upper_range[0] \
                        and self.lower_range[1] < rgb_data[y][x][1] < self.upper_range[1] \
                        and self.lower_range[2] < rgb_data[y][x][2] < self.upper_range[2]:
                    lane_x.append(x)

            if len(lane_x) == 0:
                # no lane found, execute the previous control with a decaying factor

                self.vehicle.control.throttle *= 1
                self.vehicle.control.steering *= 0.98
                self.logger.info(f"No Lane found, executing discounted prev command: {self.vehicle.control}")
                return self.vehicle.control

            # if lane is found
            avg_x = int(np.average(lane_x))

            # find error
            center_x = rgb_data.shape[1] // 2

            error = avg_x - center_x
            # we want small error to be almost ignored, only big errors matter.

            for e, scale in self.error_scaling:
                if abs(error) <= e:
                    error = error * scale
                    break

            self.kwargs["lat_error"] = error
            self.vehicle.control = self.controller.run_in_series()
            return self.vehicle.control
        else:
            # image feed is not available yet
            return VehicleControl()
