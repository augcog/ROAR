import socket
import struct
from ROAR.utilities_module.module import Module
import logging
from typing import Optional, List, Dict


class UDPMulticastCommunicator(Module):
    def save(self, **kwargs):
        pass

    def __init__(self, agent, mcast_group: str = "224.1.1.1", mcast_port: int = 5004, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.logger = logging.getLogger("UDPMulticastCommunicator")
        self.MCAST_GRP = mcast_group
        self.MCAST_PORT = mcast_port
        self.ttl = 2  # 2-hop restriction in the network
        self.recv_sock = None
        self.connect()
        self.send_sock = socket.socket(socket.AF_INET,
                                       socket.SOCK_DGRAM,
                                       socket.IPPROTO_UDP)
        self.send_sock.setsockopt(socket.IPPROTO_IP,
                                  socket.IP_MULTICAST_TTL,
                                  self.ttl)
        self.msg_log: Dict[str, List[float]] = dict()
        self.counter = 0

    def run_in_series(self, **kwargs):
        try:
            self.counter += 1
            if self.counter % 100 == 0:
                self.reconnect()
                self.counter = 0
            data: bytes = self.recv_sock.recv(1024)
            msg = data.decode('utf-8').split(",")
            self.msg_log[msg[0]] = [float(i) for i in msg[1:]]
            # print(self.msg_log.get("car_1"))
        except socket.timeout as e:
            pass
        except Exception as e:
            self.logger.error(e)

    def reconnect(self):
        try:
            print("reconnecting")
            self.recv_sock.close()
            self.connect()
        except:
            pass

    def connect(self):
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mreq = struct.pack("4sl", socket.inet_aton(self.MCAST_GRP), socket.INADDR_ANY)
        self.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.recv_sock.bind(('', self.MCAST_PORT))

    def send_msg(self, msg: str):
        self.send_sock.sendto(msg.encode('utf-8'), (self.MCAST_GRP, self.MCAST_PORT))

    def send_current_state(self):
        self.send_msg(f"{self.name},{','.join(map(str, self.agent.vehicle.to_array()))}")
