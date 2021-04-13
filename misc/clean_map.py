import numpy as np
import cv2


def saveNumpyAsImage(numpy_path="easy_map_global_occu_map.npy", output_path="map.png"):
    cv2.namedWindow('image')
    image = np.load(numpy_path).astype(np.float)
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(output_path, image)
    print(f"map written to [{output_path}]")


def saveImageAsNumpy(image_path="map_cleaned.png", output_path="easy_map_cleaned_global_occu_map.npy"):
    image = (cv2.imread(image_path)[:, :, 0] / 255).astype(np.float32)
    np.save(output_path, image)
    print(f"map written to [{output_path}]")


if __name__ == "__main__":
    # saveNumpyAsImage() # at this point, use your favorite image editing software to manually draw on it. use black and white ONLY!
    saveImageAsNumpy()
