from pydantic import Field, BaseModel
from pathlib import Path
from ROAR.utilities_module.camera_models import Camera
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
import os
from datetime import date


class Configuration(BaseModel):
    # ROAR sensors settings
    name: str = Field(default="hero", title="Name of the agent", description="Duplicate from Carla Setting. "
                                                                             "But good to have")
    front_depth_cam: Camera = Field(default=Camera(fov=70,
                                                   transform=Transform(
                                                       location=Location(x=1.6,
                                                                         y=0,
                                                                         z=1.7
                                                                         ),
                                                       rotation=Rotation(
                                                           pitch=0,
                                                           yaw=0,
                                                           roll=0)),

                                                   image_size_x=800,
                                                   image_size_y=600),
                                    title="Front Depth Camera")
    front_rgb_cam: Camera = Field(default=Camera(fov=70,
                                                 transform=Transform(
                                                     location=Location(x=1.6,
                                                                       y=0,
                                                                       z=1.7),
                                                     rotation=Rotation(pitch=0,
                                                                       yaw=0,
                                                                       roll=0)
                                                 ),
                                                 image_size_x=800,
                                                 image_size_y=600),
                                  title="Front RGB Camera")
    rear_rgb_cam: Camera = Field(default=Camera(fov=145,
                                                transform=Transform(
                                                    location=Location(x=-1.5,
                                                                      y=0.0,
                                                                      z=1.4),
                                                    rotation=Rotation(
                                                        pitch=0.0, yaw=180,
                                                        roll=0.0)),

                                                image_size_x=800,
                                                image_size_y=600),
                                 title="Rear RGB Camera")
    # data path
    waypoint_file_path: str = Field(default=(Path(
        os.getcwd()) / "data" / "berkeley_minor_waypoints.txt").as_posix())

    json_waypoint_file_path: str = Field(default=(Path(
        os.getcwd()) / "data" / "berkeley_minor_waypoints.json").as_posix())

    json_qr_code_file_path: str = Field(default=(Path(
        os.getcwd()) / "ROAR_Jetson" / "data" / "track_1.json"
                                                 ).as_posix())

    output_data_folder_path: str = Field(
        default=(Path(os.getcwd()) / "data" / "output"), description="path to save newly generated waypoints txt file.")

    output_data_file_name: str = Field(default="map_waypoints", description="file name for a newly generated waypoints txt file.")             
    
    # miscellaneous settings
    spawn_point_id: int = Field(default=1, title="Spaning Location ID",
                                description="Spanning Location ID")
    show_sensors_data: bool = Field(default=False)
    save_sensor_data: bool = Field(default=False)
    absolute_maximum_map_size: int = Field(
        default=1000, title="Absolute Maximum size of the map",
        description="This variable is used to intialize the Occupancy grid map."
                    "The bigger it is, the more impact it is going to have on the runtime computation"
                    "However, if it is smaller than the actual map, some weird things can happen")
    enable_autopilot: bool = Field(default=True, title="Enable Antopilot",
                                   description="Enable Antopilot")
    num_laps: int = Field(default=1, description="Number of laps to run for")
    max_speed: float = Field(default=200, description="maximum speed in which the vehicle can drive at") 
    target_speed: int = Field(default=80, description="The tracking speed that the pid controller is trying to achieve")
    steering_boundary: tuple = Field(default=(-1,1), description="maximum and minimum boundary for steering") # ROAR Academy:
    throttle_boundary: tuple = Field(default=(0,1), description="maximum and minimum boundary for steering") # ROAR Academy:
    waypoints_look_ahead_values: dict = Field(default={}, description="If empty, WLAV will be set to data read from \
                                                                        simple_waypoint_local_planner_config_file_path configuration") # ROAR Academy
    simple_waypoint_local_planner_config_file_path: str = \
        Field(default="./ROAR_Sim/configurations/simple_waypoint_local_planner_config.json")
    global_pid_values: dict = Field(default=None, description="If None, pid is set to data read from pid_config_file_path configuration") # ROAR Academy 
    regional_pid_values: dict = Field(default={}, description="If empty or current_region not in keys, global_pid_values will be used.") # ROAR Academy
    pid_config_file_path: str = Field(default="./ROAR_Sim/configurations/pid_config.json", description="If global_pid_values is set, \
                                                                                                        this is not read and not used.")
    lqr_config_file_path: str = Field(default="./ROAR_Sim/configurations/lqr_config.json")
