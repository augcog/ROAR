from ROAR_Jetson.jetson_runner import JetsonRunner
from ROAR.utilities_module.vehicle_models import Vehicle
from ROAR_Jetson.configurations.configuration import Configuration as JetsonConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
# from ROAR.agent_module.floodfill_based_lane_follower import FloodfillBasedLaneFollower
# from ROAR.agent_module.pid_agent import PIDAgent
# from ROAR.agent_module.jetson_pure_pursuit_agent import PurePursuitAgent
from ROAR.agent_module.special_agents.waypoint_generating_agent import WaypointGeneratigAgent
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent
from pathlib import Path
import logging
import warnings
import numpy as np
import os
import json
import sys
import serial
import argparse


def main():
    try:
        agent_config = AgentConfig.parse_file(Path("./ROAR_Jetson/configurations/agent_configuration.json"))
        jetson_config = JetsonConfig.parse_file(Path("./ROAR_Jetson/configurations/configuration.json"))

        try:
            prepare(jetson_config=jetson_config)
        except Exception as e:
            logging.error(f"Ignoring Error during setup: {e}")
        agent = ForwardOnlyAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=False)
        jetson_runner = JetsonRunner(agent=agent, jetson_config=jetson_config)
        jetson_runner.start_game_loop(use_manual_control=True)
    except Exception as e:
        print(f"Something bad happened {e}")


def prepare(jetson_config: JetsonConfig):
    if 'win' in sys.platform:
        # windows, just detect whether arduino exist on COM4
        s = serial.Serial("COM5")
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, help="debug flag", type=str2bool)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG if args.debug is True else logging.INFO)
    logging.getLogger("Vive Tracker Client [tracker_1]").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.simplefilter("ignore")
    np.set_printoptions(suppress=True)
    main()
