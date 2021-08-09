from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
import cv2
import numpy as np

class RealtimePlotterAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ensure recording status is ON
        # self.agent_settings.save_sensor_data = True
        super().__init__(**kwargs)
        self.logger = logging.getLogger("Recording Agent")

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(RealtimePlotterAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)
        self.transform_history.append(self.vehicle.transform)
        # print(self.vehicle.transform)

        return VehicleControl()
