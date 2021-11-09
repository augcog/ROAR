from pydantic import BaseModel, Field
import numpy as np
from scipy.spatial import distance
from typing import Union, Optional
from typing import List
from ROAR.utilities_module.utilities import rotation_matrix_from_euler


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
        return f"x: {round(self.x, 3)}, y: {round(self.y, 3)}, z: {round(self.z,3)}"

    def __truediv__(self, scalar):
        return Location(x=self.x / scalar, y=self.y / scalar, z=self.z / scalar)

    def to_array(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    def to_string(self) -> str:
        return f"{self.x},{self.y},{self.z}"

    @staticmethod
    def from_array(array):
        return Location(x=array[0], y=array[1], z=array[2])


class Rotation(BaseModel):
    pitch: float = Field(..., title="Pitch", description="Degree around the Y-axis")
    yaw: float = Field(..., title="Yaw", description="Degree around the Z-axis")
    roll: float = Field(..., title="Roll", description="Degree around the X-axis")

    def __str__(self):
        return f"R: {round(self.roll, 3)}, P: {round(self.pitch, 3)}, Y: {round(self.yaw, 3)}"

    def to_array(self) -> np.array:
        return np.array([self.pitch, self.yaw, self.roll])

    @staticmethod
    def from_array(array):
        return Rotation(pitch=array[0], yaw=array[1], roll=array[2])

    def __add__(self, other):
        """"""
        return Rotation(pitch=self.pitch + other.pitch, yaw=self.yaw + other.yaw, roll=self.roll + other.roll)

    def __truediv__(self, scalar):
        return Rotation(pitch=self.pitch / scalar, yaw=self.yaw / scalar, roll=self.roll / scalar)

    def __rmul__(self, scalar):
        return Rotation(pitch=self.pitch * scalar, yaw=self.yaw * scalar, roll=self.roll * scalar)

    __mul__ = __rmul__


class Transform(BaseModel):
    location: Location = Field(default=Location(x=0, y=0, z=0))
    rotation: Rotation = Field(default=Rotation(pitch=0, yaw=0, roll=0))

    def get_matrix(self) -> np.ndarray:
        """
        Calculate extrinsics matrix with respect to parent object
        http://planning.cs.uiuc.edu/node104.html

        Returns:
            Extrinsics matrix

        [R, T]
        [0 1]
        """
        location = self.location
        rotation = self.rotation

        roll, pitch, yaw = rotation.roll, rotation.pitch, rotation.yaw
        rotation_matrix = rotation_matrix_from_euler(roll=roll, pitch=pitch, yaw=yaw)

        matrix = np.identity(4)
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0:3, 0:3] = rotation_matrix
        return matrix

    def __str__(self):
        return f"Location: {self.location.__str__()} | Rotation: {self.rotation.__str__()}"

    def record(self):
        return f"{self.location.x},{self.location.y},{self.location.z},{self.rotation.roll},{self.rotation.pitch},{self.rotation.yaw}"

    def to_array(self) -> np.ndarray:
        return np.array([self.location.x, self.location.y, self.location.z, self.rotation.roll, self.rotation.pitch,
                         self.rotation.yaw])

    @staticmethod
    def from_array(array):
        return Transform(location=Location.from_array(array[:3]), rotation=Rotation.from_array(array[3:]))

    def __add__(self, other):
        return Transform.from_array(self.to_array() + other.to_array())

    def __truediv__(self, scalar):
        return Transform.from_array(self.to_array() / scalar)

    def __rmul__(self, scalar):
        return Transform.from_array(self.to_array() * scalar)

    __mul__ = __rmul__

    def readStr(self, raw: str, delimeter: str = ","):
        self.location.x, self.location.y, self.location.z, self.rotation.roll, self.rotation.pitch, \
        self.rotation.yaw = [float(val) for val in raw.split(delimeter)]

    @staticmethod
    def fromBytes(raw: bytes):
        t = Transform()
        t.readStr(raw=raw.decode("utf-8"), delimeter=",")
        return t


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


class TrackingData(BaseModel):
    transform: Transform = Field(default=Transform())
    velocity: Vector3D = Field()


class SensorsData(BaseModel):
    front_rgb: Optional[RGBData] = Field(default=None)
    rear_rgb: Optional[RGBData] = Field(default=None)
    front_depth: Optional[DepthData] = Field(default=None)
    imu_data: Optional[IMUData] = Field(default=None)
    location: Optional[Location] = Field(default=None)
    rotation: Optional[Rotation] = Field(default=None)
    velocity: Optional[Vector3D] = Field(default=None)


class MapEntry(BaseModel):
    point_a: List[float]
    point_b: List[float]
