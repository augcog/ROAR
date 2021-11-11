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
        self.should_brake = False

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if self.front_depth_camera.data is not None:
            cv2.imshow("depth", self.front_depth_camera.data)
            cv2.waitKey(1)

        if self.should_brake:
            return VehicleControl(throttle=-0.1, steering=0)
        else:
            if abs(self.vehicle.get_speed(self.vehicle)) > 8:
                self.should_brake = True
                return VehicleControl(throttle=-0.1, steering=0)
            return VehicleControl(throttle=0.4, steering=0)
