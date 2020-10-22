import socket
import json
from model import ViveTrackerMessage
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8192  # The port used by the server


def json_message(direction):
    data = ViveTrackerMessage()

    json_data = json.dumps(data.json(), sort_keys=False, indent=2)
    print("data %s" % json_data)

    send_message(json_data + ";")

    return json_data


def send_message(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.encode())
        data = s.recv(1024)

    print('Received', repr(data))


json_message("SOME_DIRECTION")
