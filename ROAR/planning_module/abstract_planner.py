from abc import ABC, abstractmethod
import logging
from typing import Any


class AbstractPlanner(ABC):
    def __init__(self, agent):
        self.logger = logging
        self.logger = logging.getLogger(__name__)
        self.agent = agent

    @abstractmethod
    def run_step(self) -> Any:
        """
        On every step, produce an actionable plan
        Returns:
        """
        return None

