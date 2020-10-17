from ROAR.planning_module.abstract_planner import AbstractPlanner
from typing import Any


class BehaviorPlanner(AbstractPlanner):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)

    def run_in_series(self) -> Any:
        pass
