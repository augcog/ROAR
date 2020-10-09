"""
This file defines a basic Bridge for extensibility of the ROAR Autonomous
software

"""
from abc import abstractmethod, ABC
import logging
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import (
    Location,
    Rotation,
    RGBData,
    DepthData,
    SensorsData,
    IMUData,
    Vector3D,
    Transform,
)
from typing import Any


class Bridge(ABC):
    def __init__(self):
        self.logger = logging.Logger(__name__)

    """ 
        INPUT 
    """

    @abstractmethod
    def convert_location_from_source_to_agent(self, source) -> Location:
        pass

    @abstractmethod
    def convert_rotation_from_source_to_agent(self, source) -> Rotation:
        pass

    @abstractmethod
    def convert_transform_from_source_to_agent(self, source) -> Transform:
        pass

    @abstractmethod
    def convert_control_from_source_to_agent(self, source) -> VehicleControl:
        pass

    @abstractmethod
    def convert_rgb_from_source_to_agent(self, source) -> RGBData:
        pass

    @abstractmethod
    def convert_depth_from_source_to_agent(self, source) -> DepthData:
        pass

    @abstractmethod
    def convert_vector3d_from_source_to_agent(self, source) -> Vector3D:
        pass

    @abstractmethod
    def convert_imu_from_source_to_agent(self, source) -> IMUData:
        pass

    @abstractmethod
    def convert_sensor_data_from_source_to_agent(self, source) -> SensorsData:
        pass

    @abstractmethod
    def convert_vehicle_from_source_to_agent(self, source) -> Vehicle:
        pass

    """ 
        OUTPUT 
    """

    @abstractmethod
    def convert_control_from_agent_to_source(self, control: VehicleControl) -> Any:
        pass

    @abstractmethod
    def convert_vector3d_from_agent_to_source(self, vector3d: Vector3D) -> Any:
        pass
