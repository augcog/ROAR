from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path


class WaypointGeneratigAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings, **kwargs)
        self.output_file_path: Path = self.output_folder_path / "easy_map_waypoints.txt"
        if self.output_folder_path.exists() is False:
            self.output_folder_path.mkdir(exist_ok=True, parents=True)
        self.output_file = self.output_file_path.open('w')


    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(WaypointGeneratigAgent, self).run_step(sensors_data=sensors_data,
                                                     vehicle=vehicle)
        if self.time_counter > 1:
            print(f"Writing to [{self.output_file_path}]: {self.vehicle.transform}")
            self.output_file.write(self.vehicle.transform.record() + "\n")
        return VehicleControl()
