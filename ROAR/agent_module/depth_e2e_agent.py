from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from pathlib import Path
from tensorflow import keras
import tensorflow as tf
import numpy as np
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class DepthE2EAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_file_path: Path = Path("./ROAR_Sim/data/weights/depth_model_1.h5")
        print(model_file_path.exists(), model_file_path)
        self.model: keras.models.Model = keras.models.load_model(model_file_path.as_posix())

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(DepthE2EAgent, self).run_step(sensors_data, vehicle)
        control = VehicleControl(throttle=0.5, steering=0)
        if self.front_depth_camera.data is not None:
            depth_image = self.front_depth_camera.data.copy()
            data = np.expand_dims(np.expand_dims(depth_image, 2), 0)
            output = self.model.predict(data)
            throttle, steering = output[0]
            control.throttle, control.steering = float(throttle), float(steering)
            print(steering)
        return control
