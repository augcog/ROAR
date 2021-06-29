from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_Sim.carla_client.util.utilities import CarlaCarColor, CarlaCarColors

class PitStop:
    def __init__(self, carla_config: CarlaConfig, agent_config: AgentConfig):
        self.carla_config = carla_config
        self.agent_config = agent_config

    """
    Set maximum speed in which the vehicle can drive at.

    Examples:
        speed = 100
    """
    def set_max_speed (self, speed: float=100):
        self.agent_config.max_speed = speed
        
    """
    Set maximum speed in which the vehicle can drive at.

    Examples:
        speed = 100
    """
    def set_target_speed (self, speed: float=40):
        self.agent_config.target_speed = speed

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
        z     : left light on
        x     : right light on
        q     : both lights on
        h     : end 
        tab   : car view change
        1 ~ 9 : Camera Type

    """
    def set_autopilot_mode(self, enabled: bool=True):
        self.agent_config.enable_autopilot = enabled
    
    """
    Set Carla Version.

    Tips:
        If carla does not open normally, try different version number.
    
    Examples:
        version = "0.9.10" 
        version = "0.9.9" 
    """
    def set_carla_version (self, version: str="0.9.9"):
        self.carla_config.carla_version = version 

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

    Examples:
    """
    # def set_simple_waypoint_local_planner_values(self,):


    # def set_pid_values(self, ):

    # def set_lqr_values(self, ):
    
   
# class AgentPITSTOP(pidAgent: PIDAgent):
#      # PIDAgent # add these to config files
#     target_speed = 40
#     steering_boundary = (-1, 1)
#     throttle_boundary = (0, 1)
#     # 6 pid values
#     longitudinal_controller = {
        
#     }
#     # waypoint
#     simple_waypoint_local_planner_config = {
#         "60": 5, 
#         "80": 10, 
#         "120": 20, 
#         "180": 50
#     }