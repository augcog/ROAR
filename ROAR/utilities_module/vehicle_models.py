from pydantic import BaseModel, Field
from typing import Optional
import math
from ROAR.utilities_module.data_structures_models import Transform, Vector3D
import numpy as np


class VehicleControl(BaseModel):
    throttle: float = Field(default=0)
    steering: float = Field(default=0)
    brake: bool = Field(default=False)

    @staticmethod
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)

    def get_throttle(self) -> float:
        """
        Cap it between -1  and 1
        :return:
        """
        return self.clamp(self.throttle, -1, 1)

    def get_steering(self) -> float:
        """
        Cap it between -1  and 1
        :return:
        """
        return self.clamp(self.steering, -1, 1)

    def to_array(self) -> np.ndarray:
        return np.array([self.throttle, self.steering])

    def record(self):
        return f"{self.brake},{round(self.throttle, 3)},{round(self.steering, 3)}"

    def __str__(self):
        return f"Brake: {self.brake}, Throttle: {round(self.throttle, 3)}, Steering: {round(self.steering, 3)}"

    @staticmethod
    def fromBytes(data: bytes):
        data = data.decode('utf-8').split(',')
        return VehicleControl(throttle=float(data[0]), steering=float(data[1]))


class Vehicle(BaseModel):
    """
    Encodes the Vehicle's state at the last tick
    """

    velocity: Optional[Vector3D] = Field(default=Vector3D(x=0, y=0, z=0))
    transform: Optional[Transform] = Field(default=Transform())
    acceleration: Optional[Vector3D] = Field(default=Vector3D(x=0, y=0, z=0))
    control: VehicleControl = Field(default=VehicleControl())  # ?
    wheel_base: float = Field(
        default=2.875,
        title="Wheel Base length of the vehilce in meters",
        description="Default to tesla model 3's wheel base",
    )

    @staticmethod
    def get_speed(vehicle):
        # TODO consider the jetson case
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        vel = vehicle.velocity
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.transform.to_array(), self.velocity.to_array(), self.acceleration.to_array(),
                               self.control.to_array()])

    def __repr__(self):
        return f"Location: {self.transform.location.__str__()} | Rotation: {self.transform.rotation.__str__()} | " \
               f"Velocity: {self.velocity.__str__()} | Acceleration: {self.acceleration.__str__()}"
