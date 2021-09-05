from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time
from typing import Tuple
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
import logging


class PotentialFieldPlanner(LoopSimpleWaypointFollowingLocalPlanner):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.logger = logging.getLogger("PotentialFieldPlanner")
        self.occu_map: OccupancyGridMap = self.agent.occupancy_map
        self._view_size = 50
        self.KP = 5.0  # attractive potential gain
        self.ETA = 2  # repulsive potential gain
        # the number of previous positions used to check oscillations
        self.OSCILLATIONS_DETECTION_LENGTH = 3
        self.AREA_WIDTH = 0
        self.step_actions = np.array([[0, -1], [-1, -1], [1, -1]])

    def is_done(self):
        return False

    def run_in_series(self) -> VehicleControl:
        # super(PotentialFieldPlanner, self).run_in_series()
        goal_world = self.find_next_waypoint()
        try:
            m = self.occu_map.get_map(transform=self.agent.vehicle.control, vehicle_value=-10,
                                      view_size=(self._view_size, self._view_size))
            goal_world_loc = goal_world.location
            curr_location = self.agent.vehicle.control.location
            gx, gy = goal_world_loc.x - curr_location.x, goal_world_loc.y - curr_location.y

            obstacle_coords = np.where(m > 0.5)
            me_coord = np.where(m == -10)
            sx, sy = me_coord[0][0], me_coord[1][0]
            gx, gy = np.clip(self._view_size // 2 + gx, 0, self._view_size - 1), \
                     np.clip(0 + gy, 0, self._view_size - 1)
            ox, oy = obstacle_coords

            rx, ry = self.potential_field_planning(sx=sx, sy=sy, gx=gx, gy=gy, ox=ox, oy=oy,
                                                   reso=1, rr=1,
                                                   world_size=m.shape)
        except Exception as e:
            self.logger.error(e)
        # waypoints = np.array(list(zip(rx, ry)))
        # # m = m.copy()
        # # for x, y in waypoints:
        # #     print(x, y)
        # x, y = rx[-1], ry[-1]
        # waypoint = self.occu_map.cropped_occu_to_world(
        #     cropped_occu_coord=np.array([x, y]),
        #     vehicle_transform=self.agent.vehicle.transform,
        #     occu_vehicle_center=np.array([self._view_size // 2, self._view_size // 2])
        # )
        control = self.controller.run_in_series(next_waypoint=goal_world)
        return control

    def potential_field_planning(self,
                                 sx, sy, gx, gy, ox, oy, reso, rr, world_size):

        # calc potential field
        # print(len(ox))
        # calc potential field
        pmap = self.calc_potential_field(gx, gy, ox, oy, reso, rr, sx, sy,
                                         world_size=world_size)

        # search path
        d = np.hypot(sx - gx, sy - gy)
        ix = sx  # where i am now
        iy = sy  # where i am now
        rx, ry = [sx], [sy]
        previous_ids = deque()

        count = 0
        while d > rr and count < 30:
            inx, iny = self.find_curr_min_action(pmap, ix, iy, step_size=1)
            if inx > len(pmap) or iny > len(pmap[0]) or inx < 0 or iny < 0:
                # searched a point that is outside of the map
                break
            else:
                ix, iy = inx, iny
            d = np.hypot(gx - ix, gy - iy)
            rx.append(ix)
            ry.append(iy)
            if self.oscillations_detection(previous_ids, ix, iy):
                break
            count += 1

        self.draw_heatmap(pmap, rx, ry)
        return rx, ry

    def find_curr_min_action(self, world, ix, iy, step_size=2) -> Tuple[int, int]:
        vals = [world[iy + action[1]][ix + action[0]] for action in self.step_actions * step_size]
        min_i = np.argmin(vals)
        min_a = self.step_actions[min_i]
        world_ix, world_iy = np.array([ix, iy]) + min_a
        return world_ix, world_iy

    def calc_potential_field(self, gx, gy, ox, oy, reso, rr, sx, sy, world_size):
        world = np.zeros(shape=world_size)
        world = self.calc_attractive_potential_vec(world=world, gx=gx, gy=gy).T
        world = self.calc_repulsive_potential_vec(world=world, ox=ox, oy=oy, rr=rr)
        return world

    def calc_repulsive_potential_vec(self, world: np.ndarray, ox: np.ndarray, oy: np.ndarray, rr) -> np.ndarray:
        if len(ox) == 0:
            return world
        else:
            o_s = 2
            for x, y in zip(ox, oy):
                # increase potential value of points around obstacles
                world[x - o_s:x + o_s, y - o_s:y + o_s] += 0.5 * self.ETA * (1 / 1.13 - 1 / rr) ** 2
                world[x][y] += 0.5 * self.ETA * (1 / 0.1 - 1 / rr) ** 2

            # obstacle_coords = np.array(list(zip(ox, oy)))
            # indices = indices.reshape((2, indices.shape[1] * indices.shape[2])).T
            # for x, y in indices:
            #     val = self.calc_repulsive_potential(x, y, obstacle_coords, rr=rr)
            #     world[x][y] += val
            return world

    def calc_attractive_potential_vec(self, world: np.ndarray, gx, gy, KP=5, res=1):
        indices = np.indices(world.shape)
        world = 0.5 * KP * np.hypot(indices[0, :, :] - gx, indices[1, :, :] - gy)
        return world

    def calc_attractive_potential(self, x, y, gx, gy):
        return 0.5 * self.KP * np.hypot(x - gx, y - gy)

    def calc_repulsive_potential(self, x, y, obstacle_coords, rr):
        # search nearest obstacle
        if len(obstacle_coords) == 0:
            return 0.0
        distances: np.ndarray = np.hypot(obstacle_coords[:, 0] - x, obstacle_coords[:, 1] - y)
        dq = distances.min()

        if dq <= rr:
            if dq <= 0.1:
                dq = 0.1

            return 0.5 * self.ETA * (1.0 / dq - 1.0 / rr) ** 2
        else:
            return 0.0


    def oscillations_detection(self, previous_ids, ix, iy):
        previous_ids.append((ix, iy))

        if len(previous_ids) > self.OSCILLATIONS_DETECTION_LENGTH:
            previous_ids.popleft()

        # check if contains any duplicates by copying into a set
        previous_ids_set = set()
        for index in previous_ids:
            if index in previous_ids_set:
                return True
            else:
                previous_ids_set.add(index)
        return False

    @staticmethod
    def draw_heatmap(data: np.ndarray, rx=None, ry=None):
        heatmapshow = None
        heatmapshow = cv2.normalize(data, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

        if rx is not None and ry is not None:
            for x, y, in zip(rx, ry):
                heatmapshow[int(y)][int(x)] = (255, 255, 255)
        cv2.imshow("heatmap", cv2.resize(heatmapshow, dsize=(500, 500)))
        cv2.waitKey(1)
