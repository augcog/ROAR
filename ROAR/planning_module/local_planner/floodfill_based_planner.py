from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.vehicle_models import VehicleControl


class FloodfillBasedPlanner(LocalPlanner):
    def run_in_series(self) -> VehicleControl:
        pass