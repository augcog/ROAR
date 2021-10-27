import logging
from pathlib import Path
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent
# from ROAR.agent_module.michael_pid_agent import PIDAgent
# from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
# from ROAR.agent_module.free_space_auto_agent import FreeSpaceAutoAgent
import argparse
from misc.utils import str2bool
from ROAR.utilities_module.utilities import NvidiaModel


def main(args):
    """Starts game loop"""
    agent_config = AgentConfig.parse_file(Path("./ROAR/configurations/carla/carla_agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("./ROAR_Sim/configurations/configuration.json"))
    carla_runner = CarlaRunner(carla_settings=carla_config,
                               agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = ForwardOnlyAgent(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=not args.auto)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", type=str2bool, default=False, help="True to use auto control")

    main(parser.parse_args())
