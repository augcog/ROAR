from typing import Any, Tuple

from Bridges.bridge import Bridge
from ROAR.utilities_module.data_structures_models import Vector3D, SensorsData, \
    IMUData, DepthData, RGBData, Transform, Rotation, Location, ViveTrackerData, TrackingData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import numpy as np
import cv2
from typing import Optional
from ROAR_Jetson.vive.models import ViveTrackerMessage
from ROAR_Jetson.jetson_vehicle import Vehicle as JetsonVehicle
from ROAR_Jetson.camera_d_t import RealsenseD435iAndT265
from typing import Union


class JetsonBridge(Bridge):
    def convert_location_from_source_to_agent(self, source:np.ndarray) -> Location:
        """
        Convert Location data from Jetson Vehicle to Agent's location data type.
        Args:
            source ():

        Returns:
            Location(x, y, z)
        """
        return Location(x=source[0], y=source[1], z=source[2])

    def convert_rotation_from_source_to_agent(self, source: np.ndarray) -> Rotation:
        """
        Convert a Jetson raw rotation to Rotation(pitch=float,yaw=float,roll=float).
        Args:
            source ():

        Returns:
            Rotation(pitch, yaw, roll)
        """
        return Rotation(roll=source[0], pitch=source[1], yaw=source[2])

    def convert_transform_from_source_to_agent(self, source) -> Transform:
        """
        Convert Jetson raw location and rotation to Transform(location,rotation).
        Args:
            source ():

        Returns:
            Transform(Location, Rotation)
        """
        return Transform(
            location=self.convert_location_from_source_to_agent(source=source.location),
            rotation=self.convert_rotation_from_source_to_agent(source=source.rotation),
        )

    def convert_control_from_source_to_agent(self, source: JetsonVehicle) -> VehicleControl:
        """
        Convert Jetson raw vehicle control to VehicleControl(throttle,steering).
        Args:
            source ():

        Returns:
            VehicleControl(Throttle, Steering)
        """
        return VehicleControl(throttle=source.throttle, steering=source.steering)

    def convert_rgb_from_source_to_agent(self, source) -> Optional[RGBData]:
        """
        Convert Jetson raw Image to an Optional with RGB numpy array.
        Args:
            source ():

        Returns:
            RGBData
        """
        if source is not None:
            return RGBData(data=source)
        return None

    def convert_depth_from_source_to_agent(self, source) -> Optional[DepthData]:
        """
        Convert Jetson raw Image to an Optional with Depth numpy array.
        Args:
            source ():

        Returns:
            DepthData
        """
        if source is not None:
            depth_data = DepthData(data=source / np.amax(source))
            return depth_data
        return None

    def convert_vector3d_from_source_to_agent(self, source) -> Vector3D:
        """
        Convert Jetson raw Vector3d Data to a Vector3D Object.
        Args:
            source ():

        Returns:
            Vector3D(x, y, z)
        """
        return Vector3D(x=source.x, y=source.y, z=source.z)

    def convert_imu_from_source_to_agent(self, source) -> IMUData:
        """
        Convert Jetson raw IMUData to IMUData(accelerometer, gyroscope).
        Args:
            source ():

        Returns:
            IMUData(accelerometer, gyroscope)
        """
        # TODO fill in data here
        return IMUData(
            accelerometer=Vector3D(x=0, y=0, z=0),
            gyroscope=Vector3D(x=0, y=0, z=0),
        )

    def convert_sensor_data_from_source_to_agent(self, source) -> SensorsData:
        """
        Returns Jetson Sensors Data from raw front RGB, rear RGB, front depth, and IMU Data.
        Args:
            source ():

        Returns:
            SensorData(front_RGB, rear_RGB, front_depth, IMU_Data)
        """
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
            location=self.convert_location_from_source_to_agent(
                source=source.get("location", None)),
            rotation=self.convert_rotation_from_source_to_agent(
                source=source.get("rotation", None)),
            velocity=self.convert_location_from_source_to_agent(
                source=source.get("velocity", None))
        )

    def convert_vive_tracker_data_from_source_to_agent(self, source: Optional[ViveTrackerMessage]) -> \
            Optional[ViveTrackerData]:
        """
        Converts raw Vive Tracker data to ViveTrackerData(Location, Rotation, Velocity).
        Args:
            source ():

        Returns:
            ViveTrackerData(Location, Rotation, Velocity)
        """
        if source is not None:
            vive_tracker_data = ViveTrackerData(
                location=Location(
                    x=-source.x,
                    y=source.y,
                    z=-source.z
                ),
                rotation=Rotation(
                    roll=-source.roll,
                    pitch=source.pitch - 90,  # 不知道为什么有60度的误差
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
        """
        Converts Transform, Velocity, Wheel_Base, and Control of JetsonVehicle.
        Args:
            source ():

        Returns:
            Vehicle(Transform, Velocity, Wheel_Base, Control)
        """
        return Vehicle(wheel_base=0.26,
                       control=self.convert_control_from_source_to_agent(source=source)
        )

    def convert_control_from_agent_to_source(self, control: VehicleControl) -> Tuple:
        """
        Converts control to Throttle and Steering numpy arrays bound between -1 and 1.
        Args:
            control ():

        Returns:
            Tuple
        """
        return np.clip(control.throttle, a_min=-1, a_max=1), np.clip(control.steering, a_min=-1, a_max=1)

    def convert_vector3d_from_agent_to_source(self, vector3d: Vector3D) -> Any:
        """
        Convert Jetson Vector3D Object to Vector3D data.
        Args:
            vector3d ():

        Returns:
            Array
        """
        return [vector3d.x, vector3d.y, vector3d.z]
