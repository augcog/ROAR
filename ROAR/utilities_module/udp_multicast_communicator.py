import socket
import struct
from ROAR.utilities_module.module import Module
import logging


class UDPMulticastCommunicator(Module):
    def save(self, **kwargs):
        pass

    def __init__(self, mcast_group: str = "224.1.1.1", mcast_port: int = 5004, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger("UDPMulticastCommunicator")
        self.MCAST_GRP = mcast_group
        self.MCAST_PORT = mcast_port
        self.ttl = 2  # 2-hop restriction in the network
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mreq = struct.pack("4sl", socket.inet_aton(self.MCAST_GRP), socket.INADDR_ANY)
        self.recv_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.recv_sock.bind(('', self.MCAST_PORT))
        print("recv socket binded")

        self.send_sock = socket.socket(socket.AF_INET,
                                       socket.SOCK_DGRAM,
                                       socket.IPPROTO_UDP)
        self.send_sock.setsockopt(socket.IPPROTO_IP,
                                  socket.IP_MULTICAST_TTL,
                                  self.ttl)

    def run_in_series(self, **kwargs):
        try:
            print("socket is listening")
            print(self.recv_sock.recv(1024))
        except socket.timeout as e:
            pass
        except Exception as e:
            self.logger.error(e)

    def send_msg(self, msg: str):
        self.send_sock.sendto(msg.encode('utf-8'), (self.MCAST_GRP, self.MCAST_PORT))
