import logging
from pathlib import Path

from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner

from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR_Sim.carla_client.util.utilities import CarlaCarColor, CarlaCarColors

# Agents
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent 
from ROAR.agent_module.pid_agent import PIDAgent # immediately crashes
from ROAR.agent_module.occu_map_demo_driving_agent import OccuMapDemoDrivingAgent # Occupancy Map Demo agent. Autonomous driving not supported yet.
from ROAR.agent_module.occupancy_map_agent import OccupancyMapAgent # immediately crashes
from ROAR.agent_module.rl_pid_agent import RLPIDAgent # for rl agent training using Gym and Stable baseline.
# from ROAR.agent_module.rl_testing_pid_agent import RLTestingAgent # rl pid agent demo, requires stable_baselines

from PitStop import PitStop as PitStop

def main():
    """Starts game loop"""
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))
    agent_config = AgentConfig.parse_file(Path("./ROAR_Sim/configurations/agent_configuration.json"))

    """
    Pit Stop:
        Use different kinds of 'set' functions at PitStop to tune/fix your own car!
    """
    pitstop = PitStop(carla_config, agent_config)
    pitstop.set_carla_sync_mode(False)
    pitstop.set_autopilot_mode(True)
    pitstop.set_car_color(CarlaCarColor(r = 255,g = 200,b = 10,a = 255))
    pitstop.set_carla_version(version = "0.9.9")
    pitstop.set_max_speed(speed = 77)
    pitstop.set_target_speed(speed = 40)
    pitstop.set_steering_boundary(boundary = (-0.1, 0.1))
    pitstop.set_throttle_boundary(boundary = (0, 0.5))
    pitstop.set_waypoints_look_ahead_values(values={"10":50, "20": 50, "30": 50, "50": 50, "60": 50, "70": 50, "80": 50, "100": 50, "120": 50, "150": 50, "180": 50})
    pid_value = {
                    'longitudinal_controller': {
                        '40': {
                            'Kp': 0.1, 'Kd': 0.2, 'Ki': 0
                        }, 
                        '60': {
                            'Kp': 0.1, 'Kd': 0.2, 'Ki': 0
                        }, 
                        '150': {
                            'Kp': 0.1, 'Kd': 0.1, 'Ki': 0.1
                        }
                    },   
                    'latitudinal_controller': {
                        '60': {
                            'Kp': 0.1, 'Kd': 0.1, 'Ki': 0.1
                        }, 
                        '100': {
                            'Kp': 0.1, 'Kd': 0.2, 'Ki': 0.1
                        }, 
                        '150': {
                            'Kp': 0.1, 'Kd': 0.2, 'Ki': 0.1
                        }
                    }
                }
    pitstop.set_pid_values(pid_value)




    carla_runner = CarlaRunner(carla_settings=carla_config, # ROAR Academy: fine
                               agent_settings=agent_config, # ROAR Academy: fine
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()
        # agent = RecordingAgent(vehicle=my_vehicle, agent_settings=agent_config)
        # agent = ForwardOnlyAgent(vehicle=my_vehicle, agent_settings=agent_config)
        agent = PIDAgent(vehicle=my_vehicle, agent_settings=agent_config)
        # agent = OccuMapDemoDrivingAgent(vehicle=my_vehicle, agent_settings=agent_config)
        # agent = PIDAgent(vehicle=my_vehicle, agent_settings=agent_config)
        # agent = RLPIDAgent(vehicle=my_vehicle, agent_settings=agent_config)

        carla_runner.start_game_loop(agent=agent, use_manual_control=False)
        # carla_runner.start_game_loop(agent=agent, use_manual_control=True) # for recording_agent
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
