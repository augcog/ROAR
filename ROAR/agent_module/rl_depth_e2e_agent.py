from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl


class RLDepthE2EAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = LoopSimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5
        )

        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self, threaded=True)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        self.add_threaded_module(self.occupancy_map)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(RLDepthE2EAgent, self).run_step(sensors_data, vehicle)
        self.local_planner.run_in_series()
        option = "obstacle_coords"  # ground_coords, obstacle_coords
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update_async(points)
        if self.kwargs.get("control") is None:
            return VehicleControl()
        else:
            return self.kwargs.get("control")
