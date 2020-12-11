from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
from typing import Any
import cv2

class CascadeObjDetector(Detector):
    def __init__(self, agent: Agent, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60), **kwargs):
        """

        @param agent:
        @param scaleFactor:
        @param minNeighbors:
        @param minSize:
        @param kwargs:
        """
        super().__init__(agent)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        # use the trained cascade classifier to classify cars
        self.cascade_car = cv2.CascadeClassifier('ROAR_Sim/Cascade/cascade_train/cascade.xml')

    def run_in_series(self, **kwargs) -> Any:
        """

        @param kwargs:
        @return:
        """
        if self.agent.front_rgb_camera.data is not None:
            rgb_img = self.agent.front_rgb_camera.data.copy()
            # generate grayed images
            greyed_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            # generate and draw bounding boxes to the detected cars
            rectangles = self.cascade_car.detectMultiScale(greyed_image, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors, minSize=self.minSize)
            detection_image = self.draw_rectangles(greyed_image, rectangles)
            # visualize the rectangles detected for debugging
            cv2.imshow("grey", detection_image)
            cv2.waitKey(1)
        return rectangles

    def run_in_threaded(self, **kwargs):
        """

        @param kwargs:
        """
        pass

    def draw_rectangles(self, img, rectangles):
        """

        @param img:
        @param rectangles:
        @return:
        """
        # these colors are actually BGR
        line_color = (255, 255, 0)
        line_type = cv2.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv2.rectangle(img, top_left, bottom_right, line_color, lineType=line_type)

        return img


