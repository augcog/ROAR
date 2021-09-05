from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap


class OccuDebugAgent(Agent):
    def __init__(self, target_speed=40, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.logger = logging.getLogger("PID Agent")
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
            closeness_threshold=1)
        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self, threaded=True)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        self.add_threaded_module(self.occupancy_map)

        self.logger.debug(
            f"Waypoint Following Agent Initiated. Reading f"
            f"rom {self.route_file_path.as_posix()}")

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(OccuDebugAgent, self).run_step(vehicle=vehicle,
                                             sensors_data=sensors_data)
        self.transform_history.append(self.vehicle.transform)
        self.local_planner.run_in_series()
        option = "obstacle_coords"  # ground_coords, point_cloud_obstacle_from_depth
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update_async(points)
        control = self.kwargs.get("control", VehicleControl())
        return control
