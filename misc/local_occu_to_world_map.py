from pathlib import Path
import numpy as np
import cv2
import glob
import os
from scipy import sparse


def load_meta_data(f_path: Path) -> np.ndarray:
    assert f_path.exists(), f"{f_path} does not exist"
    return np.load(f_path.as_posix())


def create_global_occu_map(meta_data: np.ndarray, local_occu_map_dir_path: Path, regex: str) -> np.ndarray:
    assert local_occu_map_dir_path.exists(), f"{local_occu_map_dir_path} does not exist"
    min_x, min_y, max_x, max_y, map_additiona_padding = meta_data
    x_total = max_x - min_x + 2 * map_additiona_padding
    y_total = max_y - min_y + 2 * map_additiona_padding
    curr_map = np.zeros(shape=(x_total, y_total),
                        dtype=np.float16)
    file_paths = sorted(glob.glob((local_occu_map_dir_path.as_posix() + regex)), key=os.path.getmtime)
    for fpath in file_paths:
        data = sparse.load_npz(fpath).toarray()
        # data = np.load(fpath)
        curr_map = np.logical_or(data, curr_map)
        visualize(curr_map)

    return curr_map


def visualize(m: np.ndarray, wait_key=1):
    m = np.float32(m)
    cv2.imshow("map", cv2.resize(m, dsize=(500, 500)))
    cv2.waitKey(wait_key)


def save_meta(data: np.ndarray, meta_data_file_path: Path):
    np.save(meta_data_file_path.as_posix(), data)
    print(f"Meta Data saved at {meta_data_file_path}")


if __name__ == "__main__":
    meta_data_folder_path = Path("../data/output_oct_10/occupancy_map/")
    meta_data_file_path = meta_data_folder_path / "meta_data.npy"
    # save_meta(data=np.array([-1500, -1500, 1500, 1500, 40]), meta_data_file_path=Path("../ROAR_Sim/data/berkeley_minor_occu_map_meta_data.npy"))

    try:
        meta_data: np.ndarray = load_meta_data(meta_data_file_path)
        global_occu_map = create_global_occu_map(meta_data, meta_data_folder_path, regex="/frame_*.npz")
        # path = Path("../ROAR_Sim/data/berkeley_minor_global_occu_map.npy")
        # np.save(path.as_posix(), global_occu_map)
        # print(f"Global Occu map saved at [{path}]. Press any key to exit")
        visualize(global_occu_map, wait_key=0)
    except Exception as e:
        print(e)
        # meta_data = np.array([-550, -550, 550, 550, 40])
        # np.save(meta_data_file_path.as_posix(), meta_data)
        # print(f"Meta data {meta_data} Saved")
