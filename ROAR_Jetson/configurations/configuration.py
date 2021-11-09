from pydantic import BaseModel, Field


class Configuration(BaseModel):
    jetson_sudo_password_file_path: str = Field(default="configurations/jetson_sudo_setup.json")
    pygame_display_width: int = Field(default=800)
    pygame_display_height: int = Field(default=600)
    initiate_pygame: bool = Field(default=True)
    client_ip: str = Field(default="192.168.50.205")
    win_serial_port: str = Field(default="COM3")
    unix_serial_port: str = Field(default="/dev/ttyACM0")
    baud_rate: int = Field(default=9600)
    arduino_timeout: float = Field(default=0.5)
    write_timeout: float = Field(default=0.5)
    motor_max: int = Field(default=1800)
    motor_min: int = Field(default=1200)
    motor_neutral: int = Field(default=1500)
    theta_max: int = Field(default=2000)
    theta_min: int = Field(default=1000)
    command_throttle: int = Field(default=0)
    command_steering: int = Field(default=1)

    vive_tracker_host: str = Field(default="127.0.0.1")
    vive_tracker_port: int = Field(default=8000)
    vive_tracker_name: str = Field(default="Tracker")

    agent_config_path: str = Field(default="./ROAR_Jetson/configurations/agent_configuration.json")
    use_vive_tracker: bool = Field(default=False)
    use_d435i: bool = Field(default=True)
    use_t265: bool = Field(default=True)
    use_arduino: bool = Field(default=True)

    throttle_reversed: str = Field(default=False)



