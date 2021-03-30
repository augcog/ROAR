from abc import ABC, abstractmethod
import logging
from typing import Any
from ROAR.utilities_module.module import Module


class AbstractPlanner(Module):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging
        self.logger = logging.getLogger(__name__)
        self.agent = agent

    @abstractmethod
    def run_in_series(self, **kwargs) -> Any:
        """
        On every step, produce an actionable plan
        Returns:
        """
        return None

    def run_in_threaded(self, **kwargs):
        pass

    def save(self, **kwargs):
        pass