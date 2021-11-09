from pathlib import Path
from glob import glob
import os
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_txt(path: Path) -> List[float]:
    data = path.open('r').readline()
    x, y, z, _, _, _ = data.split(sep=",")
    return [x, y, z]


def generate_track(root_dir: Path, regex: str = "/*") -> np.ndarray:
    assert root_dir.exists(), "root dir does not exist"
    file_paths = sorted(glob(root_dir.as_posix() + regex), key=os.path.getmtime)
    track = []
    for fpath in file_paths:
        try:
            data = read_txt(Path(fpath))
            track.append(data)
        except Exception as e:
            print(f"Skipping {fpath}: {e}.")

    track = np.array(track, dtype=np.float64)
    return track


def visualize_track(track: np.ndarray):
    plt.scatter(track[:, 0], track[:, 2])
    plt.show()


def save_track_as_txt(track: np.ndarray, output_file_path: Path):
    output_file = output_file_path.open('w')
    for x, y, z in track:
        output_file.write(f"{x},{y},{z}\n")
    output_file.close()
    print(f"[{len(track)}] waypoints written to [{output_file_path}]")


if __name__ == "__main__":
    track = generate_track(root_dir=Path("../output_oct_10/transform"), regex="/*")
    # visualize_track(track=track)
    save_track_as_txt(track=track, output_file_path=Path("./rfs_waypoints_4.txt"))
