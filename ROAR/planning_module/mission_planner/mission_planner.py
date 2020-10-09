from typing import List
import logging
from ROAR.utilities_module.data_structures_models import Transform
from collections import deque
from ROAR.planning_module.abstract_planner import AbstractPlanner


class MissionPlanner(AbstractPlanner):
    def __init__(self, agent):
        super().__init__(agent=agent)
        self.logger = logging.getLogger(__name__)
        self.mission_plan: deque = deque()

    def run_step(self) -> List[Transform]:
        """
        Abstract run step function for Mission Planner

        Args:
            vehicle: new vehicle state

        Returns:
            Plan for next steps

        """
        return []
