from abc import ABC, abstractmethod
from ROAR.agent_module.agent import Agent
import logging
from typing import Any
from ROAR.utilities_module.module import Module

class Detector(Module):
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.logger = logging.getLogger("Base Detector")

    @abstractmethod
    def run_in_series(self, **kwargs) -> Any:
        return None

    def run_in_threaded(self, **kwargs):
        pass