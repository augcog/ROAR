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


class LineFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.debug = True
        self.line_detector = LineDetector(agent=self, threaded=True)
        self.rgb_pixel_planner = RGBPixelPlanner(agent=self, threaded=True)
        self.pid_controller = SimplePIDController(agent=self)
        self.show_depth = True

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        mask = self.line_detector.run_in_series()

        if self.show_depth and self.front_depth_camera.data is not None:
            depth_data = -1*np.log(self.front_depth_camera.data)
            depth_data = depth_data / np.max(depth_data)
            print(self.front_depth_camera.intrinsics_matrix)
            cv2.imshow("Log of Depth", depth_data)
            cv2.waitKey(1)

        if mask is not None:
            self.kwargs["lat_error"] = self.rgb_pixel_planner.run_in_series(lane_mask=mask)
            control = self.pid_controller.run_in_series()
            return control

        return VehicleControl()
