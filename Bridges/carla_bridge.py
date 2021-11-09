import carla
from carla import ColorConverter as cc
from ROAR_Sim.carla_client.util.sensors import IMUSensor
from Bridges.bridge import Bridge
from typing import Union
from ROAR.utilities_module.vehicle_models import (
    VehicleControl,
    Vehicle,
)
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

from ROAR.utilities_module.utilities import png_to_depth
import numpy as np
import cv2
from ROAR.utilities_module.utilities import rotation_matrix_from_euler


class CarlaBridge(Bridge):
    def convert_location_from_source_to_agent(self, source: carla.Location) -> Location:
        """w
        Convert Location data from Carla.location to Agent's location data type
        invert the Z axis to make it into right hand coordinate system
        Args:
            source: carla.location

        Returns:

        """
        return Location(x=source.x, y=source.z, z=source.y)
        # return Location(x=source.x, y=source.y, z=source.z)

    def convert_rotation_from_source_to_agent(self, source: carla.Rotation) -> Rotation:
        """Convert a CARLA raw rotation to Rotation(pitch=float,yaw=float,roll=float)."""
        roll, pitch, yaw = source.roll, source.pitch, source.yaw
        if yaw <= 90:
            yaw = yaw+90
        else:
            yaw = yaw - 270
        return Rotation(roll=roll, pitch=pitch, yaw=-yaw)

    def convert_transform_from_source_to_agent(
            self, source: carla.Transform
    ) -> Transform:
        """Convert CARLA raw location and rotation to Transform(location,rotation)."""
        return Transform(
            location=self.convert_location_from_source_to_agent(source=source.location),
            rotation=self.convert_rotation_from_source_to_agent(source=source.rotation),
        )

    def convert_control_from_source_to_agent(
            self, source: carla.VehicleControl
    ) -> VehicleControl:
        """Convert CARLA raw vehicle control to VehicleControl(throttle,steering)."""

        return VehicleControl(
            throttle=-1 * source.throttle if source.reverse else source.throttle,
            steering=source.steer,
        )

    def convert_rgb_from_source_to_agent(
            self, source: carla.Image
    ) -> Union[RGBData, None]:
        """Convert CARLA raw Image to a Union with RGB numpy array."""

        try:
            source.convert(cc.Raw)
            rgb_image = self._to_rgb_array(source)
            return RGBData(data=rgb_image)
        except:
            return None

    def convert_depth_from_source_to_agent(
            self, source: carla.Image
    ) -> Union[DepthData, None]:
        """Convert CARLA raw Image to a Union with Depth numpy array."""
        try:
            array = np.frombuffer(source.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (source.height, source.width, 4))  # BGRA
            array = array[:, :, :3]  # BGR
            array = array[:, :, ::-1]  # RGB
            array = png_to_depth(array)
            return DepthData(data=array)
        except:
            return None

    def convert_vector3d_from_source_to_agent(self, source: carla.Vector3D) -> Vector3D:
        """Convert CARLA raw Vector3d Data to a Vector3D Object."""
        return Vector3D(x=source.x, y=source.y, z=source.z)

    def convert_imu_from_source_to_agent(self, source: IMUSensor) -> IMUData:
        """Convert CARLA raw IMUData to IMUData(accelerometer, gyroscope)."""
        return IMUData(
            accelerometer=Vector3D(
                x=source.accelerometer[0],
                y=source.accelerometer[1],
                z=source.accelerometer[2],
            ),
            gyroscope=Vector3D(
                x=source.gyroscope[0], y=source.gyroscope[1], z=source.gyroscope[2]
            ),
        )

    def convert_sensor_data_from_source_to_agent(self, source: dict) -> SensorsData:
        """Returns CARLA Sensors Data from raw front RGB, rear RGB, front depth, and IMU Data."""
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
        )

    def convert_vehicle_from_source_to_agent(self, source: carla.Vehicle) -> Vehicle:
        """Converts Velocity, Transform, and Control of carla.Vehicle"""
        control: VehicleControl = self.convert_control_from_source_to_agent(
            source.get_control()
        )
        # this is cheating here, vehicle does not know its own location
        transform: Transform = self.convert_transform_from_source_to_agent(
            source.get_transform()
        )
        velocity: Vector3D = self.convert_vector3d_from_source_to_agent(
            source.get_velocity()
        )
        return Vehicle(velocity=velocity, transform=transform, control=control)

    def convert_control_from_agent_to_source(
            self, control: VehicleControl
    ) -> carla.VehicleControl:
        """Converts control to carla.VehicleControl"""
        return carla.VehicleControl(
            throttle=abs(control.throttle),
            steer=control.steering,
            brake=0,
            hand_brake=False,
            reverse=True if control.throttle < 0 else False,
            manual_gear_shift=False,
            gear=1,
        )

    def convert_vector3d_from_agent_to_source(
            self, vector3d: Vector3D
    ) -> carla.Vector3D:
        """Convert Vector3D Object to a CARLA raw Vector3d Data."""
        return carla.Vector3D(x=vector3d.x, y=vector3d.y, z=vector3d.z)

    def convert_location_from_agent_to_source(self, source: Location) -> carla.Location:
        """Convert Agent's Location to a Carla.Location."""
        return carla.Location(x=source.x, y=-source.z, z=source.y)

    def convert_rotation_from_agent_to_source(self, source: Rotation) -> carla.Rotation:
        """Convert Agent's Rotation to a Carla.Rotation."""
        return carla.Rotation(pitch=source.yaw, yaw=source.pitch, roll=source.roll)

    def convert_transform_from_agent_to_source(
            self, source: Transform
    ) -> carla.Transform:
        """Convert Agent's Transform to a Carla.Transform."""
        return carla.Transform(
            location=self.convert_location_from_agent_to_source(source=source.location),
            rotation=self.convert_rotation_from_agent_to_source(source=source.rotation),
        )

    def _to_bgra_array(self, image):
        """Convert a CARLA raw image to a BGRA numpy array."""
        if not isinstance(image, carla.Image):
            raise ValueError("Argument must be a carla.sensor.Image")
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def _to_rgb_array(self, image):
        """Convert a CARLA raw image to a RGB numpy array."""
        array = self._to_bgra_array(image)
        # Convert BGRA to RGB.
        array = array[:, :, :3]
        # array = array[:, :, ::-1]
        return array
