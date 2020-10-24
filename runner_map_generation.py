import logging
from pathlib import Path
import numpy as np
import warnings
from ROAR.agent_module.legacy_agents.point_cloud_map_recording_agent import PointCloudMapRecordingAgent
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.pure_pursuit_agent import PurePursuitAgent


def main():
    agent_config = AgentConfig.parse_file(Path("../ROAR/configurations/configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("../ROAR_Sim/configurations/configuration.json"))

    carla_runner = CarlaRunner(carla_settings=carla_config, agent_settings=agent_config,
                               npc_agent_class=PurePursuitAgent)
    try:
        my_vehicle = carla_runner.set_carla_world()
        agent = PointCloudMapRecordingAgent(vehicle=my_vehicle, agent_settings=agent_config)
        carla_runner.start_game_loop(agent=agent, use_manual_control=False)
    except Exception as e:
        print("Ending abnormally: ", e)
        carla_runner.on_finish()
        logging.error(f"Hint: Might be a good idea to restart Server. ")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s '
                               '- %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    np.set_printoptions(suppress=True)

    main()
