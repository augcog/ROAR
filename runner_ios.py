import time
from pathlib import Path
from ROAR_iOS.ios_runner import iOSRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_iOS.config_model import iOSConfig
from ROAR_Unity.unity_runner import iOSUnityRunner
# from ROAR.agent_module.ios_agent import iOSAgent
# from ROAR.agent_module.free_space_auto_agent import FreeSpaceAutoAgent
from ROAR.agent_module.line_following_agent_2 import LineFollowingAgent
from ROAR.agent_module.special_agents.recording_agent import RecordingAgent
from ROAR.agent_module.traffic_light_detector_agent import TrafficLightDectectorAgent
from ROAR.agent_module.aruco_following_agent import ArucoFollowingAgent
from ROAR.agent_module.udp_multicast_agent import UDPMultiCastAgent
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent
from ROAR.utilities_module.vehicle_models import Vehicle
import logging
import argparse
from misc.utils import str2bool
from ROAR.utilities_module.utilities import get_ip
import qrcode
import cv2
import numpy as np
import socket
import json
import requests


class mode_list(list):
    # list subclass that uses lower() when testing for 'in'
    def __contains__(self, other):
        return super(mode_list, self).__contains__(other.lower())


def showIPUntilAckUDP():
    img = np.array(qrcode.make(f"{get_ip()}").convert('RGB'))
    success = False
    addr = None

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.1)

    try:
        s.bind((get_ip(), 8008))
        while success is False:
            try:
                cv2.imshow("Scan this code to connect to phone", img)
                k = cv2.waitKey(1) & 0xff
                seg, addr = s.recvfrom(1024)  # this command might timeout

                if k == ord('q') or k == 27:
                    s.close()
                    break
                addr = addr
                success = True
                for i in range(10):
                    s.sendto(b"hi", addr)
            except socket.timeout as e:
                logging.info(f"Please tap on the ip address to scan QR code. ({get_ip()}:{8008}). {e}")
    except Exception as e:
        logging.error(f"Unable to bind socket: {e}")
    finally:
        s.close()
        cv2.destroyWindow("Scan this code to connect to phone")
    return success, addr[0]


def showIPUntilAck():
    img = np.array(qrcode.make(f"{get_ip()}").convert('RGB'))
    success = False
    addr = None

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        s.bind((get_ip(), 8008))
        s.settimeout(1)
        while True:
            try:
                s.listen()

                cv2.imshow("Scan this code to connect to phone", img)
                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    s.close()
                    break
                conn, addr = s.accept()
                addr = addr[0]
                if conn:
                    s.close()
                    success = True
                    break
            except socket.timeout as e:
                logging.info(f"Please tap on the ip address to scan QR code. ({get_ip()}:{8008}). {e}")
    except Exception as e:
        logging.error(f"Unable to bind socket: {e}")
    finally:
        s.close()
        cv2.destroyWindow("Scan this code to connect to phone")
    return success, addr


def is_glove_online(host, port):
    r = requests.get(url=f"http://{host}:{port}", timeout=1)
    if r.status_code == 200:
        return True
    return False


if __name__ == '__main__':
    choices = mode_list(['ar', 'vr'])
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("-auto", action='store_true', help="Enable auto control")
    parser.add_argument("-m", "--mode", choices=choices, help="AR or VR [WARNING not implemented yet!]", default="vr")
    parser.add_argument("-r", "--reconnect", action='store_true', help="Scan QR code to attach phone to PC")
    parser.add_argument("-u", "--use_unity", type=str2bool, default=False,
                        help="Use unity as rendering and control service")
    parser.add_argument("-g", "--use_glove", help="use glove based controller by supplying its ip address!")
    args = parser.parse_args()

    try:
        agent_config_file_path = Path("ROAR/configurations/iOS/iOS_agent_configuration.json")
        ios_config_file_path = Path("ROAR_iOS/configurations/ios_config.json")
        agent_config = AgentConfig.parse_file(agent_config_file_path)
        ios_config: iOSConfig = iOSConfig.parse_file(ios_config_file_path)
        ios_config.ar_mode = True if args.mode == "ar" else False
        if args.use_glove:
            try:
                is_glove_online(args.use_glove, port=81)
                ios_config.glove_ip_addr = args.use_glove
                ios_config.should_use_glove = True
            except requests.exceptions.ConnectTimeout as e:
                print(f"ERROR. Cannot find Glove at that ip address {args.use_glove}. Shutting down...")
                exit(0)
        else:
            ios_config.should_use_glove = False

        success = False
        if args.reconnect:
            success, addr = showIPUntilAckUDP()
            if success:
                ios_config.ios_ip_addr = addr
                json.dump(ios_config.dict(), ios_config_file_path.open('w'), indent=4)
                time.sleep(2)
        if success or args.reconnect is False:
            agent = ForwardOnlyAgent(vehicle=Vehicle(), agent_settings=agent_config, should_init_default_cam=True)
            if args.use_unity:
                runner = iOSUnityRunner(agent=agent, ios_config=ios_config)
            else:
                runner = iOSRunner(agent=agent, ios_config=ios_config)
            runner.start_game_loop(auto_pilot=args.auto)
    except Exception as e:
        print(f"Something bad happened: {e}")
