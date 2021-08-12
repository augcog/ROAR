from pydantic import BaseModel, Field


class iOSConfig(BaseModel):
    ios_ip_addr: str = Field(...)
    ios_port: int = Field(...)
    world_cam_route_name: str = Field("world_cam")
    face_cam_route_name: str = Field("face_cam")
    transform_route_name: str = Field("transform")
    control_route_name: str = Field("control")
    depth_cam_route_name: str = Field("world_cam_depth")
    ar_mode: bool = Field(False)

    max_throttle: float = Field(1)
    max_steering: float = Field(1)
