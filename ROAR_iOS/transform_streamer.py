import logging
from ROAR.utilities_module.module import Module
from ROAR.utilities_module.data_structures_models import Transform
from typing import Optional, List
from pathlib import Path
from websocket import create_connection


class TransformStreamer(Module):
    def save(self, **kwargs):
        self.file.write(self.transform.record())

    def __init__(self, host: str, port: int, name: str = "transform",
                 threaded: bool = True,
                 should_record: bool = True, file_path: Path = Path("./data/transforms.txt")):
        super().__init__(threaded=threaded, name=name)
        self.logger = logging.getLogger(f"{self.name} server [{host}:{port}]")
        self.host = host
        self.port = port
        self.transform: Transform = Transform()
        self.ws = None
        self.should_record = should_record
        self.file_path: Path = file_path
        self.file = open(self.file_path.as_posix(), "a")
        self.transform_history: List[str] = []
        self.logger.info(f"{name} initialized")

    def receive(self):
        try:
            self.ws = create_connection(f"ws://{self.host}:{self.port}/{self.name}", timeout=0.1)
            result: bytes = self.ws.recv()
            try:
                self.transform = Transform.fromBytes(result)
                if self.should_record:
                    self.transform_history.append(self.transform)
            except Exception as e:
                pass
                # self.logger.error(f"Failed to parse data {e}. {result}")

        except Exception as e:
            pass
            # self.logger.error(f"Failed to get data: {e}")

    def run_in_series(self, **kwargs):
        self.receive()

    def shutdown(self):
        super(TransformStreamer, self).shutdown()
        self.file.close()
