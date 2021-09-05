from pathlib import Path
from ROAR_iOS.ios_runner import iOSRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_iOS.config_model import iOSConfig
from ROAR.agent_module.line_following_agent import LineFollowingAgent
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR.agent_module.special_agents.real_time_plotter_agent import RealtimePlotterAgent
from ROAR.utilities_module.vehicle_models import Vehicle
import logging
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class mode_list(list):
    # list subclass that uses lower() when testing for 'in'
    def __contains__(self, other):
        return super(mode_list, self).__contains__(other.lower())


if __name__ == '__main__':
    choices = mode_list(['ar', 'vr'])
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", type=str2bool, default=False, help="True to use auto control")
    parser.add_argument("-m", "--mode", choices=choices, help="AR or VR")
    args = parser.parse_args()

    try:
        agent_config = AgentConfig.parse_file(
            Path("ROAR_iOS/agent_config.json")
        )
        ios_config = iOSConfig.parse_file(
            Path("ROAR_iOS/ios_config.json")
        )

        agent = LineFollowingAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=True)
        ios_runner = iOSRunner(agent=agent, ios_config=ios_config)
        ios_runner.start_game_loop(auto_pilot=args.auto)
    except Exception as e:
        print(f"Something bad happened: {e}")
