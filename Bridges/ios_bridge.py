from typing import Any

from Bridges.bridge import Bridge
from ROAR.utilities_module.data_structures_models import Vector3D, SensorsData, IMUData, DepthData, RGBData, Transform, \
    Rotation, Location
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import numpy as np
import cv2


class iOSBridge(Bridge):
    def convert_location_from_source_to_agent(self, source) -> Location:
        pass

    def convert_rotation_from_source_to_agent(self, source) -> Rotation:
        pass

    def convert_transform_from_source_to_agent(self, source: Transform) -> Transform:
        return source

    def convert_control_from_source_to_agent(self, source: VehicleControl) -> VehicleControl:
        return source

    def convert_rgb_from_source_to_agent(self, source: np.ndarray) -> RGBData:
        return RGBData(data=source)

    def convert_depth_from_source_to_agent(self, source: np.ndarray) -> DepthData:
        return DepthData(data=source)

    def convert_vector3d_from_source_to_agent(self, source) -> Vector3D:
        pass

    def convert_imu_from_source_to_agent(self, source) -> IMUData:
        pass

    def convert_sensor_data_from_source_to_agent(self, source: dict) -> SensorsData:
        front_rgb = source.get("front_rgb", None)
        front_depth = source.get("front_depth", None)
        sensor_data = SensorsData()
        if front_rgb is not None:
            sensor_data.front_rgb = self.convert_rgb_from_source_to_agent(front_rgb)
        if front_depth is not None:
            sensor_data.front_depth = self.convert_depth_from_source_to_agent(front_depth)
        return sensor_data

    def convert_vehicle_from_source_to_agent(self, source) -> Vehicle:
        transform = source.get("transform", None)
        control = source.get("control", None)
        vehicle = Vehicle()
        if transform is not None:
            vehicle.transform = self.convert_transform_from_source_to_agent(transform)
        if control is not None:
            vehicle.control = self.convert_control_from_source_to_agent(control)
        return vehicle

    def convert_control_from_agent_to_source(self, control: VehicleControl) -> Any:
        pass

    def convert_vector3d_from_agent_to_source(self, vector3d: Vector3D) -> Any:
        pass
