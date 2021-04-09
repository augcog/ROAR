"""
Agent using purely occupancy map to drive. For demo purpose only
"""
from ROAR.agent_module.agent import Agent, AgentConfig
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
import cv2
from pathlib import Path


class OccuMapDemoDrivingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.occupancy_map = OccupancyGridMap(absolute_maximum_map_size=550,
                                              world_coord_resolution=1,
                                              occu_prob=0.99,
                                              max_points_to_convert=5000)
        occu_map_file_path = Path("./ROAR_Sim/data/easy_map_cleaned_global_occu_map.npy")
        self.occupancy_map.load_from_file(file_path=occu_map_file_path)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(OccuMapDemoDrivingAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)
        self.occupancy_map.visualize(transform=self.vehicle.transform, view_size=(200, 200))
        return VehicleControl()
