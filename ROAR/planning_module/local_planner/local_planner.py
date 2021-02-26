from abc import abstractmethod
from ROAR.planning_module.abstract_planner import AbstractPlanner
from ROAR.control_module.controller import Controller
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.mission_planner import MissionPlanner
from typing import Optional
from ROAR.utilities_module.vehicle_models import VehicleControl
from collections import deque


class LocalPlanner(AbstractPlanner):
    def __init__(
            self,
            agent,
            controller: Optional[Controller] = None,
            behavior_planner: Optional[BehaviorPlanner] = None,
            mission_planner: Optional[MissionPlanner] = None,
            **kwargs
    ):
        super().__init__(agent=agent, **kwargs)
        self.controller = (
            Controller(agent=agent) if controller is None else controller
        )
        self.behavior_planner = (
            BehaviorPlanner(agent=agent)
            if behavior_planner is None
            else behavior_planner
        )
        self.mission_planner = (
            MissionPlanner(agent=agent)
            if mission_planner is None
            else mission_planner
        )
        self.way_points_queue = deque()

    @abstractmethod
    def is_done(self):
        return False

    @abstractmethod
    def run_in_series(self) -> VehicleControl:
        return VehicleControl()
