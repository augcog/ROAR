# SENDER

import socket

group = '224.1.1.1'
port = 5004

# 2-hop restriction in network
ttl = 2

sock = socket.socket(socket.AF_INET,
                     socket.SOCK_DGRAM,
                     socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                ttl)

sock.sendto(b"sender 1", (group, port))
