from pathlib import Path
from ROAR_iOS.ios_runner import iOSRunner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_iOS.config_model import iOSConfig
from ROAR_Unity.unity_runner import iOSUnityRunner
# from ROAR.agent_module.ios_agent import iOSAgent
# from ROAR.agent_module.free_space_auto_agent import FreeSpaceAutoAgent
# from ROAR.agent_module.line_following_agent_2 import LineFollowingAgent
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

print(showIPUntilAck())