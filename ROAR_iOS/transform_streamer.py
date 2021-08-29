import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.data_structures_models import Transform
from typing import Optional, List
from pathlib import Path
from websocket import create_connection


class TransformStreamer(Module):
    def save(self, **kwargs):
        # no need to save. use Agent's saving mechanism
        pass

    def __init__(self, host: str, port: int, name: str = "transform",
                 threaded: bool = True, update_interval: float = 0.01):
        super().__init__(threaded=threaded, name=name, update_interval=update_interval)
        self.logger = logging.getLogger(f"{self.name} server [{host}:{port}]")
        self.host = host
        self.port = port
        self.transform: Transform = Transform()
        self.ws = None
        self.logger.info(f"{name} initialized")

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}", timeout=0.1)
            result: bytes = self.ws.recv()
            try:
                self.transform = Transform.fromBytes(result)
            except Exception as e:
                self.logger.error(f"Failed to parse data {e}. {result}")

        except Exception as e:
            self.logger.error(f"Failed to get data: {e}")

    def run_in_series(self, **kwargs):
        self.receive()

    def shutdown(self):
        super(TransformStreamer, self).shutdown()
