import logging
from pathlib import Path
import time
import os

from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner

from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR_Sim.carla_client.util.utilities import CarlaCarColor, CarlaCarColors

# Agents
from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent # agent for new waypoints generation
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent 
from ROAR.agent_module.pid_agent import PIDAgent
from ROAR.agent_module.occu_map_demo_driving_agent import OccuMapDemoDrivingAgent # Occupancy Map Demo agent. Autonomous driving not supported yet.
from ROAR.agent_module.occupancy_map_agent import OccupancyMapAgent
from ROAR.agent_module.rl_pid_agent import RLPIDAgent # for rl agent training using Gym and Stable baseline.
# from ROAR.agent_module.rl_testing_pid_agent import RLTestingAgent # rl pid agent demo, requires stable_baselines


from pit_stop import PitStop as PitStop

def main():
    """Starts game loop"""
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))
    agent_config = AgentConfig.parse_file(Path("./ROAR_Sim/configurations/agent_configuration.json"))

    """
    Pit Stop:
        Use different kinds of 'set' functions at PitStop to tune/fix your own car!
    """
    pitstop = PitStop(carla_config, agent_config)
    pitstop.set_carla_sync_mode(synchronized = False)
    pitstop.set_autopilot_mode(enabled = True)
    pitstop.set_car_model(car_model="vehicle.tesla.cybertruck")
    pitstop.set_car_color(color = CarlaCarColor(r = 255,g = 200,b = 00,a = 255))
    pitstop.set_num_laps(num=1)
    pitstop.set_waypoint_file_path(path=(Path(
        os.getcwd()) / "ROAR_Sim" / "data" / "berkeley_minor_waypoints.txt").as_posix())
    pitstop.set_output_data_folder_path(path="./data/output")
    pitstop.set_output_data_file_name(name=time.strftime("%Y%m%d-%H%M%S-") + "map-waypoints")
    pitstop.set_max_speed(speed = 200)
    pitstop.set_target_speed(speed = 120)
    pitstop.set_steering_boundary(boundary = (-1.0, 1.0))
    pitstop.set_throttle_boundary(boundary = (0, 1.0))
    pitstop.set_waypoints_look_ahead_values(values={
                                                    "60": 5,
                                                    "80": 10,
                                                    "120": 20,
                                                    "180": 50})
    global_pid_values = {
                    "longitudinal_controller": {
                        "40": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                        },
                        "60": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                        },
                        "150": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                            }
                    },
                    "latitudinal_controller": {
                        "60": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                        },
                        "100": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                        },
                        "150": {
                            "Kp": 0.3,
                            "Kd": 0.3,
                            "Ki": 0.3
                            }
                    }
                }
    #pitstop.set_global_pid_values(global_pid_values)
    regional_pid_values_1 = {
                    "longitudinal_controller": {
                        "40": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                        },
                        "60": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                        },
                        "150": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                            }
                    },
                    "latitudinal_controller": {
                        "60": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                        },
                        "100": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                        },
                        "150": {
                            "Kp": 0.7,
                            "Kd": 0.7,
                            "Ki": 0.7
                            }
                    }
                }
    pitstop.set_regional_pid_values(range(0,20), regional_pid_values_1)
    regional_pid_values_2 = {
                    "longitudinal_controller": {
                        "40": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                        },
                        "60": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                        },
                        "150": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                            }
                    },
                    "latitudinal_controller": {
                        "60": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                        },
                        "100": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                        },
                        "150": {
                            "Kp": 0.77,
                            "Kd": 0.77,
                            "Ki": 0.77
                            }
                    }
                }
    pitstop.set_regional_pid_values(range(21,40), regional_pid_values_2)

    """Passing configurations to Carla and Agent"""
    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()

        agent = PIDAgent(vehicle=my_vehicle, agent_settings=agent_config)
        # agent = WaypointGeneratigAgent(vehicle=my_vehicle, agent_settings=agent_config)
        
        carla_runner.start_game_loop(agent=agent, use_manual_control=False) # for PIDAgent
        # carla_runner.start_game_loop(agent=agent, use_manual_control=True) # for WaypointGeneratingAgent
    
    except Exception as e:
        logging.error(f"Something bad happened during initialization: {e}")
        carla_runner.on_finish()
        logging.error(f"{e}. Might be a good idea to restart Server")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                               '- %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.DEBUG)
    import warnings

    warnings.filterwarnings("ignore", module="carla")
    main()




