from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from pathlib import Path
import cv2


class OpenCVTensorflowObjectDetectionAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig):
        super().__init__(vehicle, agent_settings)
        folder_name = "ssd_mobilenet_v2_coco_2018_03_29"
        self.weights_folder_path = \
            Path("/home/michael/Desktop/projects/ROAR/ROAR-Sim/data/weights/")
        frozen_graph_weights_path: Path = self.weights_folder_path / 'opencv_weights_and_config' / folder_name / 'frozen_inference_graph.pb'
        frozen_graph_struct_path: Path = self.weights_folder_path / 'opencv_weights_and_config' / folder_name / 'model_structure.pbtxt'
        print(f"Path set \n{frozen_graph_weights_path}\n{frozen_graph_struct_path}")
        self.tensorflowNet = cv2.dnn.readNetFromTensorflow(
            frozen_graph_weights_path.as_posix(),
            frozen_graph_struct_path.as_posix()
        )
        print("OpenCVTensorflowObjectDetectionAgent initialized")

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(OpenCVTensorflowObjectDetectionAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)
        image = self.front_rgb_camera.data.copy()
        rows, cols, channels = image.shape
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Use the given image as input, which needs to be blob(s).
        self.tensorflowNet.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))

        # Runs a forward pass to compute the net output
        networkOutput = self.tensorflowNet.forward()

        # Loop on the outputs
        for detection in networkOutput[0, 0]:

            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                area = (right - left) * (bottom - top)

                # draw a red rectangle around detected objects
                if area < 10000:
                    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                    self.logger.debug(f"Detection confirmed. Score = {score}")
                # cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        # Show the image with a rectagle surrounding the detected objects
        cv2.imshow('Image', image)
        cv2.waitKey(1)
        return VehicleControl()
