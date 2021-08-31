from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
import numpy as np
from typing import Optional, Tuple, List
import open3d as o3d
import time, cv2
from pydantic import BaseModel, Field


class GroundPlaneDetector(Detector):
    def save(self, **kwargs):
        pass

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.settings: GroundPlaneDetectorConfiguration = GroundPlaneDetectorConfiguration.parse_file(
            self.agent.agent_settings.ground_plane_detection_config_path
        )

    def run_in_series(self, **kwargs) -> Optional[Tuple[np.ndarray, List[int]]]:
        if "point_cloud" in kwargs:
            pcd: o3d.geometry.PointCloud = kwargs["point_cloud"]
            # # remove all points that are too tall
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            ground_locs = np.where(points[:, 1] < np.mean(points[:, 1]))
            pcd.points = o3d.utility.Vector3dVector(points[ground_locs])
            pcd.colors = o3d.utility.Vector3dVector(colors[ground_locs])

            # normal estimation filtering is slow
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10),
            #                      fast_normal_computation=True)
            # normals = np.abs(np.asarray(pcd.normals))
            # flat_locs = np.where(normals[:, 1] > 0.5)
            # normals = normals[flat_locs]
            # points = points[flat_locs]
            # colors = colors[flat_locs]
            # pcd.normals = o3d.utility.Vector3dVector(normals)
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            if pcd.is_empty():
                self.logger.error("Cannot find ground")
                return None
            return pcd.segment_plane(distance_threshold=self.settings.distance_threshold,
                                     ransac_n=self.settings.ransac_n,
                                     num_iterations=self.settings.num_iterations)
        return None


class GroundPlaneDetectorConfiguration(BaseModel):
    distance_threshold: float = Field(default=0.000001)
    ransac_n: int = Field(default=3)
    num_iterations: int = Field(default=200)
    abs_min_normal: float = Field(default=0.7, description="minimum y direction normal")
