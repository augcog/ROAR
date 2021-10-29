import cv2
import numpy as np
from simple_websocket_server import WebSocketServer, WebSocket
import logging
from ROAR.utilities_module.vehicle_models import VehicleControl
import threading
frame = None
throttle = 0
steering = 0


class SimpleUnityWebsocketServer(WebSocket):
    def handle(self):
        global frame
        print(self.data)
        if frame is not None:
            frame = cv2.flip(cv2.rotate(frame, cv2.ROTATE_180), 1)
            ret, encoded_frame = cv2.imencode('.jpg', img=frame)
            self.send_message(encoded_frame)

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')


class UnityServer:
    def __init__(self, host: str, port: int):
        self.logger = logging.getLogger("Unity Streamer")
        self.host = host
        self.port = port
        self.server = WebSocketServer(self.host, self.port, SimpleUnityWebsocketServer)
        self.thread = threading.Thread(target=self.start, args=())

        self.logger.debug("Initiated")

    def get_throttle(self):
        global throttle
        return throttle

    def get_steering(self):
        global steering
        return steering

    def get_control(self):
        return VehicleControl(throttle=self.get_throttle(), steering=self.get_steering())

    def update_frame(self, new_frame):
        global frame
        frame = new_frame

    def start(self):
        self.logger.debug("Server Started")
        self.server.serve_forever()

    def shutdown(self):
        self.server.close()
        self.thread.join()

    def startAsync(self):
        self.thread.start()