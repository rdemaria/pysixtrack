import numpy as np
from .particles import Particles


def get_init_particles_for_linear_map(
        closed_orbit, p0c, d, longitudinal_coordinate,
        longitudinal_momentum, **kwargs):
    part = Particles(p0c=p0c, **kwargs)
#    part = Particles()
#    part.p0c = p0c
    part.x  = closed_orbit[0] + np.array([0.0, 1.0 * d, 0.0, 0.0, 0.0, 0.0, 0.0])
    part.px = closed_orbit[1] + np.array([0.0, 0.0, 1.0 * d, 0.0, 0.0, 0.0, 0.0])
    part.y  = closed_orbit[2] + np.array([0.0, 0.0, 0.0, 1.0 * d, 0.0, 0.0, 0.0])
    part.py = closed_orbit[3] + np.array([0.0, 0.0, 0.0, 0.0, 1.0 * d, 0.0, 0.0])

    longi   = closed_orbit[4] + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0 * d, 0.0])
    plongi  = closed_orbit[5] + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 * d])

    setattr(part, longitudinal_coordinate, longi)
    setattr(part, longitudinal_momentum, plongi)

    return part


def part_to_array(part, longitudinal_coordinate, longitudinal_momentum):
    X_array = np.empty([6, 7])
    X_array[0, :] = part.x
    X_array[1, :] = part.px
    X_array[2, :] = part.y
    X_array[3, :] = part.py
    X_array[4, :] = getattr(part, longitudinal_coordinate)
    X_array[5, :] = getattr(part, longitudinal_momentum)

    return X_array


def linearize_around_closed_orbit(
        line, closed_orbit, p0c, d, longitudinal_coordinate,
        longitudinal_momentum, **kwargs):

    part = get_init_particles_for_linear_map(
        closed_orbit, p0c, d, longitudinal_coordinate,
        longitudinal_momentum, **kwargs)
    X_init = part_to_array(
        part, longitudinal_coordinate, longitudinal_momentum)

    line.track(part)
    X_fin = part_to_array(
        part, longitudinal_coordinate, longitudinal_momentum)

    m = X_fin[:, 0] - X_init[:, 0]
    M = np.empty([6, 6])
    for j in range(6):
        M[:, j] = (X_fin[:, j + 1] - X_fin[:, 0]) / d

    X_CO = X_init[:, 0] + np.matmul(
        np.linalg.inv(np.identity(6) - M), m.T)

    return X_CO, M


def healy_symplectify(M):
    # https://accelconf.web.cern.ch/e06/PAPERS/WEPCH152.PDF
    print("Symplectifying linear One-Turn-Map...")

    print("Before symplectifying: det(M) = {}".format(np.linalg.det(M)))
    I = np.identity(6)

    S = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        ]
    )

    V = np.matmul(S, np.matmul(I - M, np.linalg.inv(I + M)))
    W = (V + V.T) / 2
    if np.linalg.det(I - np.matmul(S, W)) != 0:
        M_new = np.matmul(I + np.matmul(S, W),
                          np.linalg.inv(I - np.matmul(S, W)))
    else:
        print("WARNING: det(I - SW) = 0!")
        V_else = np.matmul(S, np.matmul(I + M, np.linalg.inv(I - M)))
        W_else = (V_else + V_else.T) / 2
        M_new = -np.matmul(
            I + np.matmul(S, W_else), np.linalg(I - np.matmul(S, W_else))
        )

    print("After symplectifying: det(M) = {}".format(np.linalg.det(M_new)))
    return M_new
