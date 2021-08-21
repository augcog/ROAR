from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
import cv2
import numpy as np

from collections import deque


class LineFollowingAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.debug = True
        self.error_queue = deque(maxlen=10)

        self.kP = 0.005
        self.kD = 0
        self.kI = 0.0001

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)

        if self.front_rgb_camera.data is not None:
            try:
                # use vision to find line, and find the center point that we are supposed to follow
                img = self.clean_image(self.front_rgb_camera.data.copy())
                Xs, Ys = np.where(img == 255)
                next_point_in_pixel = (np.average(Ys).astype(int), img.shape[0] - np.average(Xs).astype(int))

                # now that we have the center point, declare robot's position as the mid, lower of the image
                robot_point_in_pixel = (img.shape[1] // 2, img.shape[0])

                # now execute a pid control on lat diff. Since we know that only the X axis will have difference
                robot_x = robot_point_in_pixel[0]
                next_point_x = next_point_in_pixel[0]

                error = robot_x - next_point_x
                self.error_queue.append(error)
                error_dt = 0 if len(self.error_queue) == 0 else error - self.error_queue[-1]
                error_it = sum(self.error_queue)

                e_p = self.kP * error
                e_d = self.kD * error_dt
                e_i = self.kI * error_it
                lat_control = np.clip(-1 * round((e_p + e_d + e_i), 3), -1, 1)

                if self.debug:
                    cv2.circle(img,
                               center=next_point_in_pixel,
                               radius=10,
                               color=(0.5, 0.5),
                               thickness=-1)
                    cv2.circle(img,
                               center=robot_point_in_pixel,
                               radius=10,
                               color=(0.5, 0.5),
                               thickness=-1)
                    cv2.imshow("img", img)
                    cv2.waitKey(1)

                control = VehicleControl(throttle=0.175, steering=lat_control)

                print(control)
                return control
            except Exception as e:
                # self.logger.error("Unable to detect line")
                return VehicleControl()

        return VehicleControl()

    def clean_image(self, orig):
        """
        Produce a cleaned image, with line marked as white
        :return:
        """
        shape = orig.shape
        img = orig[shape[0] // 2:, :, :]
        cv2.imshow("ROI", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh", thresh)

        kernel = np.ones((10, 10), np.uint8)

        img = cv2.erode(thresh, kernel)
        img = cv2.dilate(img, kernel)

        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 150

        # your answer image
        img2 = np.zeros(output.shape)
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        return img2
