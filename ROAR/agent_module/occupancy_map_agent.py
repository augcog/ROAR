from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector
from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
import numpy as np


class OccupancyMapAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.add_threaded_module(DepthToPointCloudDetector(agent=self, threaded=True))
        self.add_threaded_module(GroundPlaneDetector(agent=self, threaded=True))

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        control = VehicleControl(throttle=0.4, steering=0)
        if self.kwargs.get("ground_coords") is not None:
            ground_coords = self.kwargs.get("ground_coords")
            # print(np.min(ground_coords, axis=0), np.max(ground_coords, axis=0))
            # print(np.shape(self.kwargs.get("ground_coords")))
        return control
