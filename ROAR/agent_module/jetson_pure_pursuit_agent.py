from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from pathlib import Path
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import \
    WaypointFollowingMissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import \
    BehaviorPlanner
from ROAR.planning_module.local_planner.tmp_jetson_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.configurations.configuration import Configuration as AgentConfig


class PurePursuitAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, target_speed=3, **kwargs):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings, **kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pure_pursuit_controller = \
            PurePursuitController(agent=self,
                                  target_speed=target_speed,
                                  look_ahead_gain=0.1,
                                  look_ahead_distance=0.1)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)

        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pure_pursuit_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=0.1)

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(PurePursuitAgent, self).run_step(sensors_data=sensors_data,
                                               vehicle=vehicle)
        vehicle_control = self.local_planner.run_in_series()

        return vehicle_control