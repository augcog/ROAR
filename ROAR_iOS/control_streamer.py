import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.vehicle_models import VehicleControl
from typing import Optional, List
from pathlib import Path
from websocket import create_connection
import requests


class ControlStreamer(Module):
    def save(self, **kwargs):
        # no need to save. use Agent's saving mechanism
        pass
    def __init__(self, host: str, port: int, name: str = "transform",
                 threaded: bool = True):
        super().__init__(threaded=threaded, name=name)
        self.logger = logging.getLogger(f"{self.name} server [{host}:{port}]")
        self.host = host
        self.port = port
        self.control_rx: VehicleControl = VehicleControl()
        self.control_tx: VehicleControl = VehicleControl()
        self.ws_tx = None
        self.ws_rx = None
        self.logger.info(f"{name} initialized")

    def send(self, vehicle_control: VehicleControl):
        try:
            self.control_tx = vehicle_control
            self.ws_tx = create_connection(f"ws://{self.host}:{self.port}/{self.name}_rx", timeout=0.1)
            self.ws_tx.send(f"{vehicle_control.throttle},{vehicle_control.steering}")
        except Exception as e:
            self.logger.error(e)

    def receive(self):
        try:
            self.ws_rx = create_connection(f"ws://{self.host}:{self.port}/{self.name}_tx")
            result: bytes = self.ws_rx.recv()
            try:
                self.control_rx = VehicleControl.fromBytes(result)
            except Exception as e:
                self.logger.error(f"Failed to parse data {e}. {result}")

        except Exception as e:
            self.logger.error(f"Failed to get data: {e}")

    def run_in_series(self, **kwargs):
        self.receive()

    def shutdown(self):
        super(ControlStreamer, self).shutdown()
