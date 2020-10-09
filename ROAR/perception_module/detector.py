from abc import ABC, abstractmethod
from ROAR.agent_module.agent import Agent
import logging
from typing import Any


class Detector(ABC):
    def __init__(self, agent: Agent):
        self.agent = agent
        self.logger = logging.getLogger("Base Detector")

    @abstractmethod
    def run_step(self) -> Any:
        return None
