import logging
from pathlib import Path
import os, sys

sys.path.append(Path(os.getcwd()).parent.as_posix())
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent
from ROAR_Gym.envs.line_detect import LineDetectAgent
from pydantic import BaseModel, Field


class PitStop(BaseModel):
    carla_config: CarlaConfig = Field(default=CarlaConfig())
    agent_config: AgentConfig = Field(default=AgentConfig())


def main():
    """Starts game loop"""
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = LineDetectAgent(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=True)
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
