from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import cv2.aruco as aruco


class ArucoFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        key = getattr(aruco, f'DICT_{5}X{5}_{250}')
        self.arucoDict = aruco.Dictionary_get(key)
        self.arucoParam = aruco.DetectorParameters_create()
        self.tracking_id = 2
        self.marker_length = 0.1  # in meters

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        result: dict = self.findArucoMarkers()  # {marker id -> bbox}
        if result:
            img = self.front_rgb_camera.data.copy()
            if self.tracking_id in result:
                bbox = result[self.tracking_id]
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(bbox, self.marker_length,
                                                                               self.front_rgb_camera.intrinsics_matrix,
                                                                               self.front_rgb_camera.distortion_coefficient)
                x, y, z = tvec[0][0]
                print(x,y,z)
                cv2.aruco.drawAxis(img,
                                   self.front_rgb_camera.intrinsics_matrix,
                                   self.front_rgb_camera.distortion_coefficient,
                                   rvec,
                                   tvec,
                                   self.marker_length)

                cv2.putText(img, f"{tvec}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2)
                cv2.putText(img, f"{rvec}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2)

                cv2.imshow("img", img)
                cv2.waitKey(1)

        return self.vehicle.control

    def findArucoMarkers(self):
        if self.front_rgb_camera.data is not None:
            img = self.front_rgb_camera.data.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bboxs, ids, rejected = aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParam,
                                                       cameraMatrix=self.front_rgb_camera.intrinsics_matrix,
                                                       distCoeff=self.front_rgb_camera.distortion_coefficient)
            log = dict()
            if ids is None:
                return log
            else:
                for i in ids:
                    log[i[0]] = bboxs
                return log
        return dict()
