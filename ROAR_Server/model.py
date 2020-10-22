from pydantic import BaseModel, Field


class ViveTrackerMessage(BaseModel):
    valid: bool = Field(default=False)
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    z: float = Field(default=0.0)
    roll: float = Field(default=0.0)
    pitch: float = Field(default=0.0)
    yaw: float = Field(default=0.0)
    device_name: str = Field(default="Tracker")
