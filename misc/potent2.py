import numpy as np
import matplotlib.pyplot as plt


def calc_potential_field(gx, gy, ox, oy, world_size, rr=2):
    obstacle_coords = np.array(list(zip(ox, oy)))
    world = np.zeros(shape=world_size)
    world = calc_attractive_potential_vec(world=world, gx=gx, gy=gy)
    world = calc_repulsive_potential_vec(world=world, ox=ox, oy=oy, rr=rr)
    return world


def calc_repulsive_potential_vec(world: np.ndarray, ox: np.ndarray, oy: np.ndarray, rr) -> np.ndarray:
    indices = np.indices(world.shape)
    if len(ox) == 0:
        return world
    else:
        obstacle_coords = np.array(list(zip(ox, oy)))
        indices = indices.reshape((2, indices.shape[1] * indices.shape[2])).T
        for x, y in indices:
            val = calc_repulsive_potential(x, y, obstacle_coords, rr=rr)
            world[x][y] += val
        return world


def calc_attractive_potential_vec(world: np.ndarray, gx, gy, KP=5, res=1):
    indices = np.indices(world.shape)
    world = 0.5 * KP * np.hypot(indices[0, :, :] - gx, indices[1, :, :] - gy)
    return world


def calc_attractive_potential(x, y, gx, gy, KP=5):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, obstacle_coords, rr=1, ETA=100):
    # search nearest obstacle
    if len(obstacle_coords) == 0:
        return 0.0
    distances: np.ndarray = np.hypot(obstacle_coords[:, 0] - x, obstacle_coords[:, 1] - y)
    dq = distances.min()

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


if __name__ == "__main__":
    ox = [0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6,
          6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10,
          11, 11, 12, 13, 14, 15, 16, 17, 18, 18, 19, 19, 20, 21, 21, 22, 22,
          23, 24, 24, 25, 25, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30,
          30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35,
          36, 37, 38]
    oy = [20, 61, 21, 20, 21, 22, 61, 20, 21, 22, 61, 21, 22, 21, 22, 61, 20,
          21, 22, 20, 21, 22, 61, 20, 21, 22, 61, 20, 21, 22, 61, 21, 22, 61,
          22, 61, 60, 61, 61, 61, 61, 61, 60, 61, 60, 61, 61, 60, 61, 60, 61,
          61, 60, 61, 60, 61, 60, 61, 41, 60, 61, 41, 60, 61, 41, 60, 61, 41,
          60, 61, 41, 60, 61, 41, 60, 61, 41, 60, 61, 41, 60, 61, 41, 60, 61,
          41, 41, 41]

    sx, sy = 50, 50
    gx, gy = 50, 100
    calc_potential_field(gx, gy, ox, oy, world_size=(100, 100))
