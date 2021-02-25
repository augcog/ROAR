from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.perception_module.cascade_obj_detector import CascadeObjDetector

class CascadeObjAgent(Agent):
    def __init__(self, **kwargs):
        """

        @param kwargs:
        """
        super().__init__(**kwargs)
        self.cascade_car = CascadeObjDetector(agent=self)

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        """

        @param vehicle:
        @param sensors_data:
        @return:
        """
        super(CascadeObjAgent, self).run_step(vehicle=vehicle,
                                              sensors_data=sensors_data)
        self.cascade_car.run_in_series()
        return self.cascade_car.run_in_series()





