from pathlib import Path
import numpy as np
import cv2
import os
from utils import flip, depth2colorjet, crop_roi

img_width, img_height = 64, 64

data_dir = Path("/data/output_oct_10")
center_depth_dir = data_dir / "front_depth"
center_depth_paths = [p for p in sorted(center_depth_dir.glob("*.npy", ), key=os.path.getmtime)]
center_rgb_dir = data_dir / "front_rgb"
center_rgb_paths = [p for p in sorted(center_rgb_dir.glob("*.png", ), key=os.path.getmtime)]
veh_states = center_rgb_dir = data_dir / "vehicle_state"
veh_states_paths = [p for p in sorted(veh_states.glob("*.npy", ), key=os.path.getmtime)]


for i in range(len(center_depth_paths)):
    depth = np.load(center_depth_paths[i])
    depth = cv2.resize(depth, dsize=(img_width, img_height))
    depth = depth2colorjet(depth)

    # rgb = cv2.imread(center_rgb_paths[i].as_posix())
    # steering_angle = np.load(veh_states_paths[i])[-2]
    # depth = crop_roi(depth, min_x=0, max_x=, min_y=, max_y=)
    cv2.imshow("depth", depth)
    # cv2.imshow("rgb", rgb)
    if cv2.waitKey(33) == ord('q'):
        break
