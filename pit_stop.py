from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_Sim.carla_client.util.utilities import CarlaCarColor, CarlaCarColors

class PitStop:
    
    def __init__(self, carla_config: CarlaConfig, agent_config: AgentConfig):
        self.carla_config = carla_config
        self.agent_config = agent_config

    """
    Set Carla Version
        
    Examples:
        version="0.9.9"
        version="0.9.10"
    """
    def set_carla_version (self, version: str="0.9.9"):
        self.carla_config.carla_version = version 
        
    """
    Set carla server frame rate and clinet frame rate to be syncrhonized
    or asynchronized.

    Example:
        synchronized = False
        synchronized = True
    """
    def set_carla_sync_mode(self, synchronized: bool=False):
        self.carla_config.synchronous_mode = synchronized

    """
    Set Autopilot mode on or off.

    Example:
        enabled = True
        enabled = False

    Carla manual control keys:
        https://carla.readthedocs.io/en/stable/simulator_keyboard_input/
        h     : end the agent
        tab   : car view change
        w     : Throttle
        S     : Brake
        A D   : Steer
        1 ~ 9 : Camera Streaming Type
        z     : left light on
        x     : right light on
        q     : both lights on

    """
    def set_autopilot_mode(self, enabled: bool=True):
        self.agent_config.enable_autopilot = enabled

    """
    Set your car's color.
    
    Examples:
        self.carla_config.car_color = CarlaCarColors.BLACK
        self.carla_config.car_color = CarlaCarColors.BLUE
        self.carla_config.car_color = CarlaCarColors.GREEN
        self.carla_config.car_color = CarlaCarColors.GREY
        self.carla_config.car_color = CarlaCarColors.RED
        self.carla_config.car_color = CarlaCarColors.WHITE
        self.carla_config.car_color = CarlaCarColor(r=20, g=10, b=255, a=255)
    """
    def set_car_color (self, color=CarlaCarColors.RED):
        self.carla_config.car_color = color
    
    """
    Set agent numer of laps
        
    Examples:
        num = 1

    """
    def set_num_laps(self, num:int=1):
        self.agent_config.num_laps = num

    """
    Set output_data_folder_path
        
    Examples:
        path = "./data/output"
    """
    def set_output_data_folder_path(self, path:str="./data/output"):
        self.agent_config.output_data_folder_path = path

    """
    Set output_data_file_name
        
    Examples:
        name = "map_waypoints"
    """
    def set_output_data_file_name(self, name:str="map_waypoints"):
        self.agent_config.output_data_file_name = name

    """
    Set maximum speed in which the vehicle can drive at.

    Examples:
        speed = 200
    """
    def set_max_speed (self, speed: float=200):
        self.agent_config.max_speed = speed
        
    """
    Set maximum speed in which the vehicle can drive at.

    Examples:
        speed = 100
    """
    def set_target_speed (self, speed: float=40):
        #  different section, differnt target speed.
        # it's global target speed for now.
        # local target speed. regional.
        # within this section of waypoints, dictionary
        # use the index of two waypoints to determine.
        # 
        self.agent_config.target_speed = speed
    
    """
    Set Steering boundary

    Examples:
    """
    def set_steering_boundary(self, boundary: tuple=(-1, 1)):
        self.agent_config.steering_boundary = boundary

    """
    Set Steering boundary

    Examples:
    """
    def set_throttle_boundary(self, boundary: tuple=(0, 1)):
        self.agent_config.throttle_boundary = boundary

    """
    Set waypoints look-ahead values.

    Tips:
        - You can add thresholds.
        - You can change values for each threshold.

    Examples:
        dict = {"50": 4, "60": 6, "70": 7, "80": 11, "100": 15, "120": 20, "150": 35, "180": 50}
    """
    def set_waypoints_look_ahead_values(self, values: dict={"60": 5, "80": 10, "120": 20, "180": 50}):
        self.agent_config.waypoints_look_ahead_values = values

    
    """
    Set 6 Pid Values.

    Examples:
        PID Values = {
                        'longitudinal_controller': {
                            '40': {
                                'Kp': 0.8, 'Kd': 0.2, 'Ki': 0
                            }, 
                            '60': {
                                'Kp': 0.5, 'Kd': 0.2, 'Ki': 0
                            }, 
                            '150': {
                                'Kp': 0.2, 'Kd': 0.1, 'Ki': 0.1
                            }
                        },   
                        'latitudinal_controller': {
                            '60': {
                                'Kp': 0.8, 'Kd': 0.1, 'Ki': 0.1
                            }, 
                            '100': {
                                'Kp': 0.6, 'Kd': 0.2, 'Ki': 0.1
                            }, 
                            '150': {
                                'Kp': 0.5, 'Kd': 0.2, 'Ki': 0.1
                            }
                        }
                    }
    """
    def set_pid_values(self, values=dict):
        self.agent_config.pid_values = values

    # def set_lqr_values(self, ):