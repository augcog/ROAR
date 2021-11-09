from abc import ABC, abstractmethod
from typing import Optional
import socketserver
import socket
import logging


class ROARServer(ABC):
    def __init__(self, port):
        self.port = port
        self.socket: Optional[socket.socket] = socket.socket(socket.AF_INET,  # Internet
                                                             socket.SOCK_DGRAM)  # UDP
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 20)
        self.logger = logging.getLogger("Base Server")

    @abstractmethod
    def run(self):
        pass

    def initialize_socket(self) -> socket.socket:
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # Some systems don't support SO_REUSEPORT
        soc.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_TTL, 20)
        soc.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_LOOP, 1)
        soc.bind(('', self.port))  # do NOT modify this line's order with other lines
        intf = socket.gethostbyname(socket.gethostname())
        soc.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(intf))
        soc.settimeout(3)
        self.socket = soc
        return soc