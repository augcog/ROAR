import logging
from pathlib import Path
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR.agent_module.potential_field_agent import PotentialFieldAgent
from ROAR.agent_module.occupancy_map_agent import OccupancyMapAgent
from ROAR.agent_module.michael_pid_agent import PIDAgent
# from ROAR.agent_module.depth_e2e_agent import DepthE2EAgent
from pydantic import BaseModel, Field


class PitStop(BaseModel):
    carla_config: CarlaConfig = Field(default=CarlaConfig())
    agent_config: AgentConfig = Field(default=AgentConfig())


def main():
    """Starts game loop"""
    agent_config = AgentConfig.parse_file(Path("./ROAR_Sim/configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))

    pit_stop = PitStop(carla_config=carla_config, agent_config=agent_config)
    pit_stop.agent_config.save_sensor_data = False
    pit_stop.agent_config.look_ahead = {
        60: 1,
        80: 2
    }
    pit_stop.agent_config.pid_config = {
        "longitudinal_controller": {
            "40": {
                "Kp": 0.8,
                "Kd": 0.4,
                "Ki": 0
            },
            "60": {
                "Kp": 0.5,
                "Kd": 0.3,
                "Ki": 0
            },
        },
        "latitudinal_controller": {
            "60": {
                "Kp": 0.8,
                "Kd": 0.1,
                "Ki": 0.1
            },
            "100": {
                "Kp": 0.6,
                "Kd": 0.2,
                "Ki": 0.1
            }
        }
    }

    carla_runner = CarlaRunner(carla_settings=pit_stop.carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = PotentialFieldAgent(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=False)
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
