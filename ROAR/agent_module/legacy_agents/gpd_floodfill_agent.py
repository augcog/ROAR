from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
import cv2


class GPDFloodFillAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, target_speed=50):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.gpd_detector = GroundPlaneDetector(self)

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        control = super(GPDFloodFillAgent, self).run_step(sensors_data=sensors_data,
                                                          vehicle=vehicle)
        self.gpd_detector.run_in_series()
        return control
