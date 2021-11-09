"""
The purpose of this file is to take in a txt file in containing data
x,y,z,roll,pitch,yaw or x,y,z
...
and visualize the track
"""
from pathlib import Path
from typing import List
import numpy as np
import plotly.graph_objects as go


def read_txt(file_path: Path) -> List[List[float]]:
    if file_path.exists() is False:
        raise FileNotFoundError(f"{file_path} is not found. Please double check")
    file = file_path.open('r')
    result: List[List[float]] = []
    for line in file.readlines():
        try:
            x, y, z = line.split(sep=',')
        except Exception as e:
            x, y, z, roll, pitch, yaw = line.split(sep=',')

        result.append([float(x), float(y), float(z)])
    return result


def visualize_track_data(track_data: List[List[float]]):
    print(f"Visualizing [{len(track_data)}] data points")
    track_data = np.asarray(track_data)
    fig = go.Figure(data=[go.Scatter3d(x=track_data[:, 0], y=[0] * len(track_data), z=track_data[:, 2],
                                       mode='markers')])
    fig.show()


if __name__ == "__main__":
    track_data: List[List[float]] = read_txt(Path("./output_oct_10/output_fast.txt"))
    visualize_track_data(track_data=track_data)
