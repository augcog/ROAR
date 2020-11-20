from typing import Any, Tuple

from Bridges.bridge import Bridge
from ROAR.utilities_module.data_structures_models import Vector3D, SensorsData, \
    IMUData, DepthData, RGBData, Transform, Rotation, Location, ViveTrackerData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import numpy as np
import cv2
from typing import Optional
from ROAR_Jetson.vive.models import ViveTrackerMessage
from ROAR_Jetson.jetson_vehicle import Vehicle as JetsonVehicle


class JetsonBridge(Bridge):
    def convert_location_from_source_to_agent(self, source) -> Location:
        return Location(x=source.x, y=source.y, z=-source.z)

    def convert_rotation_from_source_to_agent(self, source) -> Rotation:
        return Rotation(pitch=source.pitch, yaw=source.yaw, roll=source.roll)

    def convert_transform_from_source_to_agent(self, source) -> Transform:
        return Transform(
            location=self.convert_location_from_source_to_agent(source=source.location),
            rotation=self.convert_rotation_from_source_to_agent(source=source.rotation),
        )

    def convert_control_from_source_to_agent(self, source: JetsonVehicle) -> VehicleControl:
        return VehicleControl(throttle=source.throttle, steering=source.steering)

    def convert_rgb_from_source_to_agent(self, source) -> Optional[RGBData]:
        if source is not None:
            return RGBData(data=source)
        return None

    def convert_depth_from_source_to_agent(self, source) -> Optional[DepthData]:
        if source is not None:
            return DepthData(data=source)
        return None

    def convert_vector3d_from_source_to_agent(self, source) -> Vector3D:
        return Vector3D(x=source.x, y=source.y, z=source.z)

    def convert_imu_from_source_to_agent(self, source) -> IMUData:
        # TODO fill in data here
        return IMUData(
            accelerometer=Vector3D(x=0, y=0, z=0),
            gyroscope=Vector3D(x=0, y=0, z=0),
        )

    def convert_sensor_data_from_source_to_agent(self, source) -> SensorsData:
        return SensorsData(
            front_rgb=self.convert_rgb_from_source_to_agent(
                source=source.get("front_rgb", None)
            ),
            rear_rgb=self.convert_rgb_from_source_to_agent(
                source=source.get("rear_rgb", None)
            ),
            front_depth=self.convert_depth_from_source_to_agent(
                source=source.get("front_depth", None)
            ),
            imu_data=self.convert_imu_from_source_to_agent(
                source=source.get("imu", None)
            ),
            vive_tracker_data=self.convert_vive_tracker_data_from_source_to_agent(
                source=source.get("vive_tracking", None))
        )

    def convert_vive_tracker_data_from_source_to_agent(self, source: Optional[ViveTrackerMessage]) -> \
            Optional[ViveTrackerData]:
        if source is not None:
            vive_tracker_data = ViveTrackerData(
                location=Location(
                    x=-source.x,
                    y=source.y,
                    z=-source.z
                ),
                rotation=Rotation(
                    roll=-source.roll,
                    pitch=source.pitch,
                    yaw=-source.yaw
                ),
                velocity=Vector3D(
                    x=source.vel_x,
                    y=source.vel_y,
                    z=source.vel_z
                )
            )
            return vive_tracker_data
        return None

    def convert_vehicle_from_source_to_agent(self, source: JetsonVehicle) -> Vehicle:
        return Vehicle(transform=Transform(
            location=Location(x=-source.location[0], y=source.location[1], z=-source.location[2]),
            rotation=Rotation(roll=-source.rotation[2], pitch=source.rotation[0], yaw=-source.rotation[1]),
        ),
            velocity=Vector3D(x=-source.velocity[0], y=source.velocity[1], z=-source.velocity[2]),
            wheel_base=0.26,
            control=self.convert_control_from_source_to_agent(source=source)
        )

    def convert_control_from_agent_to_source(self, control: VehicleControl) -> Tuple:
        return np.clip(control.throttle, a_min=-1, a_max=1), np.clip(control.steering, a_min=-1, a_max=1)

    def convert_vector3d_from_agent_to_source(self, vector3d: Vector3D) -> Any:
        return [vector3d.x, vector3d.y, vector3d.z]
