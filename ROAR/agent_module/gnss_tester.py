from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import LidarData, SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path


class GnessTester(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings, **kwargs)


    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(GnessTester, self).run_step(sensors_data=sensors_data,
                                                     vehicle=vehicle)
        if self.gnss != None:
            print("GNSS:% 24s"% ("(% 2.3f, % 3.3f)" % (self.gnss.lat, self.gnss.lon)))

        return VehicleControl()

  
