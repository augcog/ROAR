from typing import Any

from Bridges.bridge import Bridge
from ROAR.utilities_module.data_structures_models import Vector3D, SensorsData, IMUData, DepthData, RGBData, Transform, \
    Rotation, Location
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import numpy as np
import cv2


class iOSBridge(Bridge):
    def convert_location_from_source_to_agent(self, source) -> Location:
        return Location(
            x=source.x,
            y=-source.z,
            z=source.y
        )

    def convert_rotation_from_source_to_agent(self, source: Rotation) -> Rotation:
        r = Rotation(
            roll=np.rad2deg(source.pitch),
            pitch=np.rad2deg(source.roll),
            yaw=np.rad2deg(source.yaw)
        )
        return r

    def convert_transform_from_source_to_agent(self, source: Transform) -> Transform:
        return Transform(location=self.convert_location_from_source_to_agent(source.location), rotation=self.convert_rotation_from_source_to_agent(source.rotation))

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
        velocity = source.get("velocity", None)
        control = source.get("control", None)
        acceleration = source.get("acceleration", None)
        vehicle = Vehicle()
        if transform is not None:
            vehicle.transform = self.convert_transform_from_source_to_agent(transform)
        if velocity is not None:
            vehicle.velocity = Vector3D(
                x=velocity.x,
                y=-velocity.z,
                z=velocity.y,
            )
        if control is not None:
            vehicle.control = self.convert_control_from_source_to_agent(control)
        if acceleration is not None:
            vehicle.acceleration = acceleration
        return vehicle

    def convert_control_from_agent_to_source(self, control: VehicleControl) -> VehicleControl:
        return control

    def convert_vector3d_from_agent_to_source(self, vector3d: Vector3D) -> Any:
        pass
