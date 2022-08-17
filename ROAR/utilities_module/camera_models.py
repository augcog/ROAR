from pydantic import BaseModel, Field
import numpy as np
from ROAR.utilities_module.data_structures_models import (
    Location,
    Rotation,
    Transform,
)
from typing import Optional
import cv2


class Camera(BaseModel):
    fov: int = Field(default=70, title="Field of View")
    transform: Transform = Field(
        default=Transform(
            location=Location(x=1.6, y=0, z=1.7),
            rotation=Rotation(pitch=0, yaw=0, roll=0),
        )
    )
    image_size_x: int = Field(default=800, title="Image size width")
    image_size_y: int = Field(default=600, title="Image size width")
    distortion_coefficient: Optional[np.ndarray] = Field(default=None,
                                                         title="Distortion Coefficient from Intel Realsense")
    data: Optional[np.ndarray] = Field(default=None)
    intrinsics_matrix: Optional[np.ndarray] = Field(default=None)

    def calculate_default_intrinsics_matrix(self) -> np.ndarray:
        """
        Calculate intrinsics matrix
        Will set the attribute intrinsic matrix so that re-calculation is not
        necessary.
        https://github.com/carla-simulator/carla/issues/56
        [
                ax, 0, cx,
                0, ay, cy,
                0 , 0, 1
        ]
        Returns:
            Intrinsics_matrix
        """
        intrinsics_matrix = np.identity(3)
        intrinsics_matrix[0, 2] = self.image_size_x / 2.0
        intrinsics_matrix[1, 2] = self.image_size_y / 2.0
        intrinsics_matrix[0, 0] = self.image_size_x / (
                2.0 * np.tan(self.fov * np.pi / 360.0)
        )
        intrinsics_matrix[1, 1] = self.image_size_y / (
                2.0 * np.tan(self.fov * np.pi / 360.0)
        )
        self.intrinsics_matrix = intrinsics_matrix
        return intrinsics_matrix

    class Config:
        arbitrary_types_allowed = True

    def visualize(self, title="CameraData", duration=1) -> None:
        """
        Visualize camera data.
        Args:
            title: title of cv2 image
            duration: in milisecond
        Returns:
            None
        """
        if self.data is not None:
            cv2.imshow(title, self.data.data)
            cv2.waitKey(duration)


class LidarConfigModel(BaseModel):
    channels: int = Field(32)
    range: int = Field(100)
    points_per_second: int = Field(56000)
    rotation_frequency: float = Field(10)
    upper_fov: float = Field(10)
    lower_fov: float = Field(-30)
    horizontal_fov: float = Field(360)
    atmosphere_attenuation_rate: float = Field(0.004)
    dropoff_general_rate: float = Field(0.45)
    dropoff_intensity_limit: float = Field(0.8)
    dropoff_zero_intensity: float = Field(0.4)
    sensor_tick: float = Field(0)
    noise_stddev: float = Field(0)

    transform: Transform = Field(
        default=Transform(
            location=Location(x=1.6, y=0, z=1.7),
            rotation=Rotation(pitch=0, yaw=0, roll=0),
        )
    )


