from pydantic import BaseModel, Field
import numpy as np
from scipy.spatial import distance
from typing import Union, Optional
from typing import List


class Location(BaseModel):
    x: float = Field(
        ...,
        title="X coordinate",
        description="Distance in meters from origin to spot on X axis",
    )
    y: float = Field(
        ...,
        title="Y coordinate",
        description="Distance in meters from origin to spot on Y axis",
    )
    z: float = Field(
        ...,
        title="Z coordinate",
        description="Distance in meters from origin to spot on Z axis",
    )

    def distance(self, other_location):
        """Euclidean distance between current location and other location"""
        return distance.euclidean(
            (self.x, self.y, self.z),
            (other_location.x, other_location.y, other_location.z),
        )

    def __add__(self, other):
        """"""
        return Location(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __str__(self):
        return f"x: {self.x:.3}, y: {self.y:.3}, z: {self.z:.3}"

    def to_array(self) -> np.array:
        return np.array([self.x, self.y, self.z])


class Rotation(BaseModel):
    pitch: float = Field(..., title="Pitch", description="Degree around the Y-axis")
    yaw: float = Field(..., title="Yaw", description="Degree around the Z-axis")
    roll: float = Field(..., title="Roll", description="Degree around the X-axis")

    def __str__(self):
        return f"Roll: {round(self.roll, 2)}, Pitch: {round(self.pitch, 2)}, Yaw: {round(self.yaw, 2)}"

    def to_array(self) -> np.array:
        return np.array([self.pitch, self.yaw, self.roll])


class Transform(BaseModel):
    location: Location = Field(default=Location(x=0, y=0, z=0))
    rotation: Rotation = Field(default=Rotation(pitch=0, yaw=0, roll=0))

    def get_matrix(self) -> np.ndarray:
        """
        Calculate extrinsics matrix with respect to parent object
        http://planning.cs.uiuc.edu/node104.html

        Returns:
            Extrinsics matrix
        """
        location = self.location
        rotation = self.rotation
        yaw, pitch, roll = rotation.yaw, rotation.pitch, rotation.roll
        c_y = np.cos(np.radians(yaw))
        s_y = np.sin(np.radians(yaw))
        c_r = np.cos(np.radians(roll))
        s_r = np.sin(np.radians(roll))
        c_p = np.cos(np.radians(pitch))
        s_p = np.sin(np.radians(pitch))

        matrix = np.identity(4)
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def __str__(self):
        return f"Location: {self.location.__str__()} | Rotation: {self.rotation.__str__()}"

    def record(self):
        return f"{self.location.x},{self.location.y},{self.location.z},{self.rotation.roll},{self.rotation.pitch},{self.rotation.yaw}"


class Vector3D(BaseModel):
    x: float = Field(default=0)
    y: float = Field(default=0)
    z: float = Field(default=0)

    def to_array(self):
        return np.array([self.x, self.y, self.z])


class RGBData(BaseModel):
    data: np.ndarray = Field(
        ..., title="RGB Data", description="Array of size (WIDTH, HEIGHT, 3)"
    )

    class Config:
        arbitrary_types_allowed = True


class DepthData(BaseModel):
    data: np.ndarray = Field(
        ..., title="Depth Data", description="Array of size (WIDTH, HEIGHT, 3)"
    )

    class Config:
        arbitrary_types_allowed = True


class IMUData(BaseModel):
    accelerometer: Vector3D = Field(
        default=Vector3D(),
        title="Accelerometer data",
        description="Linear acceleration in m/s^2",
    )
    gyroscope: Vector3D = Field(
        default=Vector3D(),
        title="Gyroscope data",
        description="Angular velocity in rad/sec",
    )


class ViveTrackerData(BaseModel):
    location: Location = Field(default=Location(x=0, y=0, z=0))
    rotation: Rotation = Field(default=Rotation(roll=0, pitch=0, yaw=0))
    velocity: Vector3D = Field()
    tracker_name: str = Field(default="Tracker")


class SensorsData(BaseModel):
    front_rgb: Union[RGBData, None] = Field(default=None)
    rear_rgb: Union[RGBData, None] = Field(default=None)
    front_depth: Union[DepthData, None] = Field(default=None)
    imu_data: Union[IMUData, None] = Field(default=None)
    vive_tracker_data: Optional[ViveTrackerData] = Field(default=None)


class MapEntry(BaseModel):
    point_a: List[float]
    point_b: List[float]
