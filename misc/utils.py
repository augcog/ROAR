import argparse
import cv2
import numpy as np
from typing import Tuple


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def depth2colorjet(depth: np.ndarray) -> np.ndarray:
    """
    Given a 2d numpy array of floats ranging from 0 - 1,
    turn it into a uint8 2d numpy array with jet color mapping
    """
    depth = -1 * np.log(depth)
    depth = (depth / np.max(depth) * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return rgb


def flip(img: np.ndarray, steering_angle: float) -> Tuple[np.ndarray, float]:
    return np.fliplr(img), -steering_angle


def crop_roi(img: np.ndarray, min_x: int, min_y: int, max_x: int, max_y: int):
    return img[min_x:max_x, min_y:max_y]


def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def depthToLogDepth(depth_image):
    output = np.log(depth_image)
    output = 1 - (output / np.min(output))
    return output

if __name__ == '__main__':
    from pathlib import Path
    import glob
    import os
    img_dir = Path("/data/output_oct_10/front_depth")
    img_paths = [p for p in sorted(img_dir.glob("*.npy", ), key=os.path.getmtime)]
    for img_path in img_paths:
        im = np.load(img_path.as_posix())

        cv2.imshow("depth image", im)
        cv2.imshow("log depth", depthToLogDepth(im))
        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break

