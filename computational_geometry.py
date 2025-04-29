import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# 2d Line segment y value
# ----------------------------------------------------------------------------------------------------------------------
def line_segment_y_value(x: float,
                         ep0: np.ndarray,
                         ep1: np.ndarray,
                         t_vert: float = 1.0) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Find the y value of a 2d line given x value and two endpoints
    :param x: value at which to find y
    :param ep0: x and y coordinates of an end point
    :param ep1: x and y coordinates of an end point
    :param t_vert: value at assign curve paramter when line passing through provided endpoints is vertical
    :return: tuple containing the value of y, as well as curve parameter t, where t=0 is ep1 and t=1 is ep2
    """

    # check dimensions
    scalar_flag = False
    if len(ep0.shape) == 1 and len(ep1.shape) == 1:
        ep0 = ep0[:, None]
        ep1 = ep1[:, None]
        scalar_flag = True

    # Difference in end point coordinates
    dx = (ep1[0, :] - ep0[0, :])
    dy = (ep1[1, :] - ep0[1, :])

    # Flag indicating line is vertical
    is_vert = (dx == 0)

    # Calculate t
    t = x - ep0[0, :]
    t[~is_vert] = t[~is_vert] / dx[~is_vert]
    t[is_vert] = t_vert

    # Calculate y
    y = ep0[1, :] + t * dy

    if scalar_flag:
        t = t[0]
        y = y[0]

    return y, t

# ----------------------------------------------------------------------------------------------------------------------
# Intersections
# ----------------------------------------------------------------------------------------------------------------------
def polyline_intersect_naive(xy: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Find locations where a polyline intersects itself using brute force algorithm. Time complexity is O(n^).
    :param xy: a [2xn] array of cartesian coordinate pairs representing vertices of a polyline
    :return: a [2xm] array of cartesian coordinate pairs representing intersections, and a list segment indices which contain the intersections
    """
    # Create lists
    intersections = []
    indices = []

    # compare each line segment to each other line segment
    # ith segment: pi→pi+1
    # jth segment: pj→pj+1
    # where j = [i+2, n-1] = [j, k-1]

    # Calculation performed as described here:
    # https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments

    for idx_i, _ in enumerate(xy[:, :-1].T):
        # dxi = xi1 - xi0
        # dyi = yi1 - yi0
        a = xy[0, idx_i + 1] - xy[0, idx_i]
        c = xy[1, idx_i + 1] - xy[1, idx_i]

        # compare ith segment against jth segment
        idx_k = idx_i + 2
        for idx_j, _ in enumerate(xy[:, idx_k:-1].T):
            # dxj = xj1 - xj0
            # dyj = yj1 - yj0
            b = -1 * (xy[0, idx_j + 1 + idx_k] - xy[0, idx_j + idx_k])
            d = -1 * (xy[1, idx_j + 1 + idx_k] - xy[1, idx_j + idx_k])

            # dx0ij = xj0 - xi0
            # dy0ij = yj0 - yi0
            e = xy[0, idx_j + idx_k] - xy[0, idx_i]
            f = xy[1, idx_j + idx_k] - xy[1, idx_i]

            # Lines are parallel if determinate of ((a, b), (c, d)) is zero
            if (a * d - b * c) != 0:
                # Solve for intersection as curve parameters: i.e. yt = y0 + t * (xf - x0)
                s = (1 / (a * d - b * c)) * (+1 * d * e - b * f)
                t = (1 / (a * d - b * c)) * (-1 * c * e + a * f)

                # Segments intersect if intersection is contained within both lines
                if 0 <= s <= 1 and 0 <= t <= 1:
                    intersections.append(((1 - s) * xy[0, idx_i] + s * xy[0, idx_i + 1],
                                          (1 - s) * xy[1, idx_i] + s * xy[1, idx_i + 1]))
                    indices.append((idx_i, idx_j + idx_k))

    return np.array(intersections, dtype=float).T, indices

# ----------------------------------------------------------------------------------------------------------------------
# Polyline modifications, calculations, comparisons, etc.
# ----------------------------------------------------------------------------------------------------------------------

def polyline_trim_loops(xy: np.ndarray) -> np.ndarray:
    """
    Find loops within a polyline and remove them.
    NOTE: Does not handle loops within loops
    :param xy:  a [2xn] array of cartesian coordinate pairs representing vertices of a polyline
    :return:  a [2xn] array of cartesian coordinate pairs representing vertices of the trimmed polyline
    """
    # Get intersections (slow!)
    insct_xy, insct_idx = polyline_intersect_naive(xy)

    # Remove loops by
    # 1) iterating backwards through list segments which contain intersections
    # 2) deleting columns of xy which are between the end points of the intersecting segments
    # 3) changing the end point of the first segment, which is also the start point of the last segment, to be the intersection point
    for that_idx, this_insct_idx in enumerate(reversed(insct_idx)):
        xy = np.delete(xy, slice(this_insct_idx[0] + 1, this_insct_idx[1]), 1)
        xy[:, this_insct_idx[0] + 1] = insct_xy[:, -that_idx - 1]


    return xy