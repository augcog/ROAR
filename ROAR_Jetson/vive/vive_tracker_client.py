"""
Defines Vive Tracker server. This script should run as is.

Example usage:
python vive_tracker_client.py --debug True

For Vive Tracker Server implementation, please see
https://github.com/wuxiaohua1011/ROAR_Desktop/blob/main/ROAR_Server/vive_tracker_server.py

"""
import socket
import sys
import logging
from typing import Optional

try:
    from ROAR_Jetson.vive.models import ViveTrackerMessage
except:
    from models import ViveTrackerMessage
import json
import time
from typing import Tuple
import argparse
from pathlib import Path


class ViveTrackerClient:
    """
    Defines a vive tracker client that constantly polls message from (HOST, PORT)
    and update its self.latest_tracker_message public variable

    Other interacting script can initialize this ViveTracker Client as a sub-process and access its
    latest_tracker_message for tracker data.

    multiple vive tracker can be used at the same time by initializing multiple clients with different `tracker_name`
    """

    def __init__(self, host: str, port: int, tracker_name: str,
                 time_out: float = 1, buffer_length: int = 1024,
                 should_record: bool = False,
                 output_file_path: Path = Path("../data/RFS_Track.txt")):
        """

        Args:
            host: Server's LAN Host address. (Ex: 192.168.1.7)
            port: Server's LAN Port address. (Ex: 8000)
            tracker_name: Tracker name (Ex: tracker_1)
            time_out: time out for socket's receive. Will reset socket on timeout
            buffer_length: maximum length of data it can receive at once
            should_record: enable recording of data
            output_file_path: output file's path
        """
        self.host = host
        self.port = port
        self.tracker_name: str = tracker_name
        self.time_out = time_out
        self.buffer_length = buffer_length
        self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 20)
        self.socket.settimeout(self.time_out)
        self.latest_tracker_message: Optional[ViveTrackerMessage] = None
        self.should_record = should_record
        self.output_file_path = output_file_path
        self.output_file = None
        if self.should_record:
            if self.output_file_path.parent.exists() is False:
                self.output_file_path.parent.mkdir(exist_ok=True, parents=True)
            self.output_file = self.output_file_path.open('w')
        self.count = 0
        self.logger = logging.getLogger(f"Vive Tracker Client [{self.tracker_name}]")
        self.logger.info("Tracker Initialized")

    def update(self):
        """
        This client will send to the server the name of the tracker it is requesting

        It will receive that tracker's information.

        Updates the self.latest_vive_tracker_message field

        Record  self.latest_vive_tracker_message if needed

        Returns:
            None
        """
        self.logger.info(f"Start Subscribing to [{self.host}:{self.port}] "
                         f"for [{self.tracker_name}] Vive Tracker Updates")
        while True:
            try:
                _ = self.socket.sendto(self.tracker_name.encode(), (self.host, self.port))
                data, addr = self.socket.recvfrom(self.buffer_length)  # buffer size is 1024 bytes
                parsed_message, status = self.parse_message(data.decode())
                if status:
                    self.update_latest_tracker_message(parsed_message=parsed_message)
                    if self.should_record:
                        if self.count % 10 == 0:
                            self.output_file.write(f'{self.latest_tracker_message.x},'
                                                   f'{self.latest_tracker_message.y},'
                                                   f'{self.latest_tracker_message.z},'
                                                   f'{self.latest_tracker_message.roll},'
                                                   f'{self.latest_tracker_message.pitch},'
                                                   f'{self.latest_tracker_message.yaw}\n')
                    self.count += 1
                else:
                    self.logger.error(f"Failed to parse incoming message [{data.decode()}]")
            except socket.timeout:
                self.logger.error("Timed out")
            except ConnectionResetError as e:
                self.logger.error(f"Error: {e}. Retrying")
            except OSError as e:
                pass
                # self.logger.error(e)
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                self.logger.debug(e)

    def run_threaded(self):
        pass

    def shutdown(self):
        """
        Safely shuts down the client and its connections
        Returns:

        """
        self.socket.close()
        if self.output_file is not None:
            self.output_file.close()

    def update_latest_tracker_message(self, parsed_message):
        """
        Given Vive Tracker message in JSON format, load json into dictionary format,
        parse the tracker message using PyDantic

        Assign self.latest_vive_tracker_message as the parsed result

        Args:
            parsed_message: tracker message in json format

        Returns:
            None
        """
        try:
            d = json.loads(json.loads(parsed_message))
            vive_tracker_message = ViveTrackerMessage.parse_obj(d)
            if vive_tracker_message.device_name == self.tracker_name:
                self.latest_tracker_message = vive_tracker_message
            self.logger.debug(self.latest_tracker_message)
        except Exception as e:
            self.logger.error(f"Error: {e} \nMaybe it is related to unable to parse buffer [{parsed_message}]. ")

    @staticmethod
    def parse_message(received_message: str) -> Tuple[str, bool]:
        """
        Parse the received message by ensuring that it start and end with special "handshake" characters

        Args:
            received_message: string format of the received bytes

        Returns:
            parsed received message in string and whether the parsing was successful

        """
        start = received_message.find("&")
        end = received_message.find("\r")
        if start == -1 or end == -1:
            return "", False
        else:
            return received_message[start + 1:end], True

    @staticmethod
    def initialize_socket() -> socket.socket:
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        soc.settimeout(3)
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Some systems don't support SO_REUSEPORT
        soc.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)
        soc.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)

        return soc


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
    parser.add_argument("--collect",default=False, help="debug flag", type=str2bool)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
                        datefmt="%H:%M:%S", level=logging.DEBUG if args.debug is True else logging.INFO)
    HOST, PORT = "192.168.1.19", 8000
    client = ViveTrackerClient(host=HOST, port=PORT, tracker_name="tracker_1",
                               output_file_path=Path("../data/RFS_Track.txt"),
                               should_record=args.collect)
    client.update()