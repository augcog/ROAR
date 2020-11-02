from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl

from ROAR.perception_module.depth_to_pointcloud_detector import DepthToPointCloudDetector
from pathlib import Path
import numpy as np
import cv2

class PointCloudRecordingAgent(Agent):
    def __init__(self, **kwargs):
        super(PointCloudRecordingAgent, self).__init__(**kwargs)
        """
        self.gp_pointcloud_detector = GroundPlanePointCloudDetector(agent=self,
                                                                    max_points_to_convert=10000,
                                                                    nb_neighbors=100,
                                                                    std_ratio=1)
        """
        self.depth_to_point_cloud_detector = DepthToPointCloudDetector(agent=self)
        self.pointcloud_output_folder_path = self.output_folder_path / "pointcloud_roar_sim" / "pointcloud"
        self.pointcloud_rgb_output_folder_path = self.output_folder_path / "pointcloud_roar_sim" / "rgb"
        if self.pointcloud_output_folder_path.exists() is False:
            self.pointcloud_output_folder_path.mkdir(parents=True, exist_ok=True)
        if self.pointcloud_rgb_output_folder_path.exists() is False:
            self.pointcloud_rgb_output_folder_path.mkdir(parents=True, exist_ok=True)


    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(PointCloudRecordingAgent, self).run_step(sensors_data, vehicle)
        try:
            if self.front_rgb_camera.data is not None:
                points = self.depth_to_point_cloud_detector.run_in_series()  # (N x 3)
                output_file_path = self.pointcloud_output_folder_path / f"{self.time_counter}.npy"
                np.save(output_file_path, arr=points)
                # cv2.imshow("rgb", self.front_rgb_camera.data)
                # cv2.waitKey(1)
                cv2.imwrite((self.pointcloud_rgb_output_folder_path / f"{self.time_counter}.png").as_posix(),
                            self.front_rgb_camera.data)

        except Exception as e:
            self.logger.error(f"Point cloud RunStep Error: {e}")
        finally:
            return VehicleControl()
