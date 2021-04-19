from abc import ABC, abstractmethod
import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.utilities_module.data_structures_models import Transform


class Controller(Module):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.logger = logging.getLogger("Controller")

    @abstractmethod
    def run_in_series(self, next_waypoint: Transform, **kwargs) \
            -> VehicleControl:
        """
        Abstract function for run step

        Args:
            next_waypoint: next waypoint
            **kwargs:

        Returns:
            VehicleControl
        """
        return VehicleControl()

    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass
