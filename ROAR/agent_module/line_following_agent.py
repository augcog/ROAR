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

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        mask = self.line_detector.run_in_series()

        if mask is not None:




            pass

        return VehicleControl()
