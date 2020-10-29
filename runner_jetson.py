from ROAR_Jetson.jetson_runner import JetsonRunner
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR_Jetson.configurations.configuration import Configuration as JetsonConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
# from ROAR.agent_module.floodfill_based_lane_follower import FloodfillBasedLaneFollower
from ROAR.agent_module.jetson_pid_agent import PIDAgent
# from ROAR.agent_module.jetson_pure_pursuit_agent import PurePursuitAgent
from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent
from pathlib import Path
import logging
import warnings
import numpy as np
import os
import json
import sys
import serial



def main():
    try:
        agent_config = AgentConfig.parse_file(Path("./ROAR_Jetson/configurations/agent_configuration.json"))
        jetson_config = JetsonConfig.parse_file(Path("./ROAR_Jetson/configurations/configuration.json"))

        try:
            prepare(jetson_config=jetson_config)
        except Exception as e:
            logging.error(f"Ignoring Error during setup: {e}")
        agent = PIDAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=False)
        jetson_runner = JetsonRunner(agent=agent, jetson_config=jetson_config)
        jetson_runner.start_game_loop(use_manual_control=True)
    except Exception as e:
        print(f"Something bad happened {e}")


def prepare(jetson_config: JetsonConfig):
    if 'win' in sys.platform:
        # windows, just detect whether arduino exist on COM4
        s = serial.Serial("COM4")
        status = s.isOpen()
    else:
        # assume that this is going to be a unix based system
        status = allow_dev_access(read_password(Path(jetson_config.jetson_sudo_password_file_path)))
    assert status is True, "Port not successfully opened"


def read_password(jetson_sudo_setup_f_path: Path):
    d = json.load(fp=jetson_sudo_setup_f_path.open("r"))
    return d["sudo_password"]


def allow_dev_access(pwd):
    command = 'chmod 777 /dev/ttyACM0'
    p = os.system(f'echo {pwd}|sudo -S {command}')
    return True if p == 0 else False


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s '
                               '- %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    np.set_printoptions(suppress=True)

    main()
