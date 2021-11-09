from pydantic import BaseModel, Field


class iOSConfig(BaseModel):
    ios_ip_addr: str = Field(...)
    ios_port: int = Field(...)
    world_cam_route_name: str = Field("world_cam")
    face_cam_route_name: str = Field("face_cam")
    veh_state_route_name: str = Field("veh_state")
    control_route_name: str = Field("control")
    depth_cam_route_name: str = Field("world_cam_depth")
    ar_mode: bool = Field(False)

    max_throttle: float = Field(1)
    max_steering: float = Field(1)

    steering_offset: float = Field(0)
    invert_steering: bool = Field(False)

    pygame_display_width: int = Field(1080)
    pygame_display_height: int = Field(810)

    should_display_system_status: bool = True

    should_use_glove: bool = Field(default=False)
    glove_ip_addr: str = Field(default="192.168.1.30")
