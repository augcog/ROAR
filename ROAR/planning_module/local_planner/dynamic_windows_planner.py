from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.vehicle_models import VehicleControl


class DWAPlanner(LocalPlanner):
    def is_done(self):
        pass

    def run_in_series(self) -> VehicleControl:
        pass

    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)