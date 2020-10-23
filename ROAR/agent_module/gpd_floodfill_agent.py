from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from pathlib import Path
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import \
    WaypointFollowingMissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import \
    BehaviorPlanner
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector


class GPDFloodFillAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, target_speed=50):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.gpd_detector = GroundPlaneDetector(self)

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(GPDFloodFillAgent, self).run_step(sensors_data=sensors_data,
                                               vehicle=vehicle)
        self.gpd_detector.run_in_series()
        return VehicleControl()
