from pathlib import Path
from ROAR_iOS.ios_runner import iOSRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_iOS.config_model import iOSConfig
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent

from ROAR.utilities_module.vehicle_models import Vehicle
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)

    try:
        agent_config = AgentConfig.parse_file(
            Path("ROAR_iOS/agent_config.json")
        )
        ios_config = iOSConfig.parse_file(
            Path("ROAR_iOS/ios_config.json")
        )

        agent = RecordingAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=True)
        ios_runner = iOSRunner(agent=agent, ios_config=ios_config)
        ios_runner.start_game_loop(auto_pilot=False)
    except Exception as e:
        print(f"Something bad happened: {e}")