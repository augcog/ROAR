
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl


class RLDepthE2EAgent(Agent):
    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        if self.kwargs.get("control") is None:
            return VehicleControl()
        else:
            return self.kwargs.get("control")