import time

import numpy as np
import matplotlib.pyplot as plt

import computational_geometry as cg


def main():
    test_flag = 3

    if test_flag == 0:
        test_line_segment_y_value()
    elif test_flag == 1:
        test_polyline_intersect()
    elif test_flag == 2:
        test_curvature()
    elif test_flag == 3:
        test_offsets()
    else:
        for idx in range(4):
            xy = generate_xy_data(idx, param=2)
            plt.plot(xy[0, :], xy[1, :])
        plt.axis('equal')
        plt.xlim((-2, 2))
        plt.ylim((-2, 2))

        plt.show()

def generate_xy_data(type: int=0, qty: int=128, param: float=1.0) -> np.ndarray:
    tt_ini = 0
    tt_end = param * 3.14159265
    tt_qty = qty
    tt = np.linspace(start=tt_ini, stop=tt_end, num=tt_qty)

    # Roulettes
    if type <= 3:
        # Major and minor diameters
        r_mjr = 1.0
        r_mnr = r_mjr / 8

        # Distance of traced point
        if type == 0 or type == 2:
            r_pnt = r_mnr
        else:
            r_pnt = 3.0 * r_mnr

        # Epicycloid or Epitrochoid
        if type == 0 or type == 1:
            xy = np.array((((r_mjr + r_mnr) * np.cos(tt) - r_pnt * np.cos(1 * tt * (r_mjr + r_mnr) / r_mnr)),
                           ((r_mjr + r_mnr) * np.sin(tt) - r_pnt * np.sin(1 * tt * (r_mjr + r_mnr) / r_mnr))), dtype=float)

        # Hypocycloid or Hypotrochoid
        else:
            xy = np.array((((r_mjr - r_mnr) * np.cos(tt) + r_pnt * np.cos(1 * tt * (r_mjr - r_mnr) / r_mnr)),
                           ((r_mjr - r_mnr) * np.sin(tt) - r_pnt * np.sin(1 * tt * (r_mjr - r_mnr) / r_mnr))),
                          dtype=float)

    # Arbitrary Curve
    elif type == 4:
        xy = (
            (-2, -1),
            (-0, -1),
            (4, -1),
            (1, 0),
            (5, 1),
            (-1, 0),
            (-1.5, 0.5),
            (-3, 1)
        )
        xy = np.array(xy, dtype=float).T

    # Circle
    elif type == 5:
        xy = 1 * np.array((np.cos(tt), np.sin(tt)), dtype=float)

    # Simple Line
    else:
        xy = np.array(((0, 1), (0, 1)), dtype=float)

    return xy

def generate_tr_data(type: int=0, qty: int=128, param: float=0) -> np.ndarray:
    # Generate cartesian data
    xy = generate_xy_data(type, qty, param)

    # map to polar coordinates
    tr = np.array(
        (np.arctan2(xy[1, :], xy[0, :]),
         np.hypot(xy[0, :], xy[1, :])),
        dtype=float
    )

    return tr

def test_line_segment_y_value():
    x_val = 0.1
    xy_0 = np.array((0, 0), dtype=float)
    xy_1 = np.array((4, 4), dtype=float)
    xy_2 = np.random.rand(2, 8)
    xy_3 = np.zeros_like(xy_2)

    y, t = cg.line_segment_y_value(x_val, xy_2, xy_3)

    fig_1, axs_1 = plt.subplots()

    for idx, this_x in enumerate(xy_2.T):
        that_x = xy_3[:, idx]
        axs_1.plot(
            (this_x[0], that_x[0]),
            (this_x[1], that_x[1])
        )

    axs_1.axvline(x_val, linestyle='--')
    axs_1.plot(x_val * np.ones_like(y), y, linestyle='', marker='.', markersize=10, color='k')

    plt.show()

def test_polyline_intersect():
    # Generate data
    xy = generate_xy_data(type=3, qty=2**7, param=1.5)

    # Start timer
    t_val = time.perf_counter()

    # Run bruteforce algorithm
    insct, _ = cg.polyline_intersect_naive(xy)

    # Stop timer and print
    t_val -= time.perf_counter()
    print(f"brute force runtime: {-1 * t_val:5.3f} [s]")

    # Test loop trimming
    xy_trim = cg.polyline_trim_loops(xy)

    # Handle curves with no intersections
    if insct.size == 0:
        insct = np.empty((2, 2))
        insct[:] = np.nan

    # Visualize
    plt.plot(xy[0, :], xy[1, :], linewidth=4)
    plt.plot(xy_trim[0, :], xy_trim[1, :])
    plt.plot(insct[0, :], insct[1, :], linestyle='', marker='.', markersize=10)
    plt.show()

def test_offsets():
    # Generate data
    xy = generate_xy_data(type=1, qty=2 ** 5, param=1.5)
    tr = generate_tr_data(type=1, qty=2 ** 5, param=1.5)

    off_dir = 0.1

    # Start timer
    t_val = time.perf_counter()

    # Calculate offset
    oc = cg.offset_cart(xy, off_dir)

    # Stop timer and print
    t_val -= time.perf_counter()
    print(f"cartesian offset runtime: {-1 * t_val:5.6f} [s]")

    # Start timer
    t_val = time.perf_counter()

    # Calculate offset
    op = cg.offset_polar(tr, off_dir, miter_threshold=0.5 * np.pi)
    op = op[1, :] * np.array((np.cos(op[0, :]), np.sin(op[0, :])), dtype=float)

    # Stop timer and print
    t_val -= time.perf_counter()
    print(f"polar offset runtime: {-1 * t_val:5.6f} [s]")


    plt.plot(xy[0, 0], xy[1, 0], color='tab:blue', marker='o', markersize=10, markerfacecolor='None')
    plt.plot(xy[0, :], xy[1, :], color='tab:blue', linewidth=4)
    plt.plot(oc[0, :], oc[1, :], color='tab:red', linewidth=4)
    plt.plot(op[0, :], op[1, :], color='tab:green', linewidth=2)
    plt.axis('equal')
    plt.show()

def test_curvature():
    xy = generate_xy_data(5, qty=2**8)
    k = cg.curvature_numeric(xy)

    fig_1, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].plot(xy[0, :], xy[1, :])
    axs[0].set_aspect('equal')

    axs[1].plot(k)
    axs[1].axhline(0, color='k', alpha=0.5)

    plt.show()

if __name__ == '__main__':
    main()