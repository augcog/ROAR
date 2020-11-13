from ROAR.agent_module.agent import Agent
from ROAR.perception_module.detector import Detector
from typing import Optional, Any
import time, cv2

class CascadeObjDetector(Detector):
    def __init__(self, agent: Agent, **kwargs):
        #super().__init__(agent, **kwargs)
        super().__init__(agent)
        self.cascade_car = cv2.CascadeClassifier('ROAR_Sim/Cascade/cascade_train/cascade.xml')

    def run_step(self) -> Any:
        if self.agent.front_rgb_camera.data is not None:
            rgb_img = self.agent.front_rgb_camera.data.copy()
            greyed_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            rectangles = self.cascade_car.detectMultiScale(greyed_image, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
            time.sleep(0.05)
            detection_image = self.draw_rectangles(greyed_image, rectangles)
            cv2.imshow("grey", detection_image)
            cv2.waitKey(1)
        #return self.cascade_car.step()

    def draw_rectangles(self, haystack_img, rectangles):
        # these colors are actually BGR
        line_color = (255, 255, 0)
        line_type = cv2.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv2.rectangle(haystack_img, top_left, bottom_right, line_color, lineType=line_type)

        return haystack_img


