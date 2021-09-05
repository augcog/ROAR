from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.perception_module.ground_plane_detector import GroundPlaneDetector

import open3d as o3d


class GPDAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.depth_to_pointcloud_detector = GroundPlaneDetector(agent=self)

    def run_step(self, vehicle: Vehicle, sensors_data: SensorsData) -> VehicleControl:
        super(GPDAgent, self).run_step(vehicle=vehicle, sensors_data=sensors_data)
        try:
            self.depth_to_pointcloud_detector.run_in_series()

        except Exception as e:
            print(e)
        # print(np.shape(points))
        return VehicleControl()


    @staticmethod
    def visualize_pointcloud(points):
        """

        :param points: Nx3 array
        :return:
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
