from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path


class WaypointGeneratigAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.output_file_path: Path = self.output_folder_path / "easy_map_waypoint_t.txt"
        self.output_file = self.output_file_path.open('w')

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(WaypointGeneratigAgent, self).run_step(sensors_data=sensors_data,
                                                     vehicle=vehicle)
        if self.time_counter % 20 == 0:
            self.output_file.write(self.vehicle.transform.__str__() + "\n")
        return VehicleControl()
