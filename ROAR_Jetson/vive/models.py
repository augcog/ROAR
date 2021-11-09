from pydantic import BaseModel, Field, validator
import math


class ViveTrackerMessage(BaseModel):
    valid: int = Field(default=0)
    x: float = Field(default=0)
    y: float = Field(default=0)
    z: float = Field(default=0)
    roll: float = Field(default=0)
    pitch: float = Field(default=0)
    yaw: float = Field(default=0)
    device_name: str = Field(default="tracker_1")
    vel_x: float = Field(default=0)
    vel_y: float = Field(default=0)
    vel_z: float = Field(default=0)

    def __repr__(self):
        return f"{self.device_name} -> " \
               f"x: {round(self.x, 5)}, y: {round(self.y, 5)}, z: {round(self.z, 5)} | " \
               f"pitch: {round(self.pitch, 5)}, yaw: {round(self.yaw, 5)}, roll: {round(self.roll, 5)} | " \
               f"vel_x: {round(self.vel_x, 5)}, vel_y: {round(self.vel_y, 5)}, vel_z: {round(self.vel_z, 5)}"

    def __str__(self):
        return self.__repr__()


