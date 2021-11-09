import time
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
from collections import deque


class ForwardOnlyAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.log = deque(maxlen=100)
        self.start = time.time()

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            cv2.imshow("depth", self.front_depth_camera.data)
            cv2.waitKey(1)
        return VehicleControl(throttle=0.4, steering=0)
