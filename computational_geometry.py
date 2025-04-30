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
    Find locations where a polyline intersects itself using brute force algorithm. Time complexity is O(n^2).
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
# Offsets
# ----------------------------------------------------------------------------------------------------------------------

def offset_cart(xy: np.ndarray, dist: float=1.0) -> np.ndarray:
    """
    Generate polyline offset
    :param xy: a [2xn] array of cartesian coordinate pairs, [xi, yi], representing vertices of a polyline
    :param dist: distance of offset; positive offsets to the "right", negative to the "left"
    :return: [2xn] array of cartesian coordinate pairs representing vertices of offset polyline
    """
    # Forward differences, normalized
    dx = np.diff(xy, axis=1)
    dx /= np.hypot(dx[0, :], dx[1, :])
    dx = np.concatenate((dx[:, 0][:, None], dx, dx[:, -1][:, None]), axis=1)

    # Vector normal to forward gradient, normalized
    nf = np.array((dx[1, 1:], -1 * dx[0, 1:]), dtype=float)

    # Vector normal to backward gradient, normalized
    nb = np.array((dx[1, :-1], -1 * dx[0, :-1]), dtype=float)

    # dot product of normals (dp), angle between (aa), and scaling factor (ss)
    dp = (nf * nb).sum(axis=0)
    aa = np.acos(dp)
    ss = 1 / np.cos(0.5 * aa)

    # Offset direction
    nn = nf + nb
    nn /= np.hypot(nn[0, :], nn[1, :])
    nn *= ss

    return xy + dist * nn

def offset_polar(tr: np.ndarray, dist: float=1.0, miter_threshold: float=np.inf) -> np.ndarray:
    """
    Curve offsetting algorithm for polyline represented in polar coordinates.
    Assume input curve is ordered ascending in Θ.
    Intersections of offset at interior corners or with original curve are not managed
    :param tr: a [2xn] array of polar coordinate pairs, [Θi, ri], representing vertices of a polyline
    :param dist: distance of offset; positive offsets to the "right", negative to the "left"
    :param miter_threshold: bisector angles greater than this threshold will be mitered
    :return: [2xn] array of polar coordinate pairs representing vertices of offset polyline
    """

    # Definitions
    # ----------------
    # Given a set of ordered points in polar coordinates, where the ith point is
    #   pi = [Θi, ri]
    # Where necessary, subscripts are contained within braces
    #   p{i+1} = [Θ{i+1}, r{i+1}]
    # Define a line segment with beginning and end points as
    #   Si = Bi→Ei
    # Where ends of line segments in polar coordinates are defined as
    #   Bi = pi = [Θi, ri]
    #   Ei = p{i+1} = [Θ{i+1}, r{i+1}] = B{i+1}
    # Segment i defines a triangle with edges Ai, Bi, Ci, where
    #   O: Origin
    #   Ai: O→Bi
    #   Bi: O→Ei = O→B{i+1}] = A{i+1}
    #   Ci: Bi→Ei = Si
    # Interior angles opposite edges Ai, Ci, Bi defined as ai, ci, bi, respectively

    # direction to offset; -1 is inward, toward decreasing r, and +1 is outward, toward increasing r
    dir = np.sign(dist)

    # Initial Triangle
    # ----------------
    # Side lengths Ai and Bi are known a priori.
    #   Ai = ri
    #   Bi = r(i+1) = A(i+1)
    A = tr[1, :-1]
    B = tr[1, 1:]

    # Interior angle ci can be calculated as:
    #   ci = Θ(i+1) - Θi
    #   Note: ci is permitted to be negative
    c = tr[0, 1:] - tr[0, :-1]

    # Bound c in [-π, π]
    c[c > +np.pi] -= 2 * np.pi
    c[c < -np.pi] += 2 * np.pi

    # From law of cosines, solve for C:
    #   C^2 = A^2 + B^2 - 2*A*B*cos(c)
    #   Note: sign of c is lost
    C = np.sqrt(A ** 2 + B ** 2 - 2 * A * B * np.cos(c))

    # Invert law of cosines to solve for b
    #   b = acos((B^2 - A^2 - C^2) / (-2*A*C))
    #   Note: c < 0 → b < 0
    b = np.sign(c) * np.acos((B ** 2 - A ** 2 - C ** 2) / (-2 * A * C))

    # Interior angles of triangle must equal π; solve for a:
    #   a = π - b - c
    #   Note: if c < 0 → b < 0 → a < 0
    a = np.sign(c) * np.pi - b - c

    # Angle and magnitude of offset
    # ----------------
    # The direction of offset is bisector, d, of the angle between Si and S{i+1}.
    # Note: if c < 0 → a, b, d < 0
    d = 0.5 * (a[:-1] + b[1:])
    d[dir * d > 0] -= dir * np.pi

    # A line along the bisector creates the hypotenuse of a right triangle, for which the base is collinear with Ci,
    # and its opposite side is perpendicular to Ci, with a length D, the desired offset distance.
    # The length of the hypotenuse can be solved for with trigonometry
    #   Hi = D / sin(d)
    H = abs(dist / np.sin(d))  # Does dropping the sign lose information?

    # Polar coordinates of offset point
    # ----------------
    # We seek the polar coordinates of the beginning point of the hypotenuse above
    #   Hi = Bi→Ei = pi*→p{i+1} = [Θ*, r*]→[Θ{i+1}, r{i+1}]
    # There exists a triangle Qi, Hi, Bi where
    #   Qi = O→pi*
    # if dir == -1:
    #     # The interior angle q is angle a less the bisector
    #     q = a[:-1] - d
    # else:
    #     # The interior angle q is the compliment of angle a plus the bisector
    #     q = 2 * np.pi - (a[:-1] + d)

    # Note if c < 0 → q might be less than zero
    q = d - a[:-1]

    # Bound q in [-π, π]
    q[q > +np.pi] -= 2 * np.pi
    q[q < -np.pi] += 2 * np.pi


    # Manage Extreme Angles, work in progress
    # --------------------------------
    # this_idx = np.where(abs(abs(d) - 0.5 * np.pi) > 0.1)[0] # old condition; not sure about this one
    this_idx = np.where(np.logical_or.reduce((
        abs(d) > miter_threshold,
        abs(d) < np.pi - miter_threshold
    )))[0]

    # half the bisector + π/4
    g = 0.5 * d[this_idx] + dir * 0.25 * np.pi

    # New hypotenuse and interior angle
    H[this_idx] = abs(dist / np.cos(g))
    q[this_idx] += g

    # insert an additional point
    tr = np.insert(tr, this_idx + 1, tr[:, this_idx + 1], axis=1)
    B = np.insert(B, this_idx + 0, B[this_idx])
    H = np.insert(H, this_idx + 0, H[this_idx])

    q = np.insert(q, this_idx + 0, q[this_idx] - 2 * g)

    # Bound q in [-π, π]
    q[q > +np.pi] -= 2 * np.pi
    q[q < -np.pi] += 2 * np.pi

    # Solve for offset points
    # --------------------------------
    # From law of cosines, solve for Qi, which is r*:
    # Note: sign of q is lost
    Q = np.sqrt(B[:-1] ** 2 + H ** 2 - 2 * B[:-1] * H * np.cos(q))

    # Invert law of cosines to solve for h; round away numerical errors
    h = np.sign(q) * np.acos(np.round((H ** 2 - B[:-1] ** 2 - Q ** 2) / (-2 * B[:-1] * Q), 6))

    # Calculate Θ* (reuse variable h)
    h += tr[0, 1:-1]

    # Convert to numpy array and return
    oo = np.stack((h, Q), axis=0)

    return oo

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

def curvature_numeric(xy: np.ndarray) -> np.ndarray:
    """
    Find curvature of a polyline numerically
    :param xy: a [2xn] array of cartesian coordinate pairs representing vertices of a polyline
    :return: an [n] element vector containing numeric value polyline curvature
    """

    # line parameter
    dt = ((xy[0, 1:] - xy[0, :-1]) ** 2 + (xy[1, 1:] - xy[1, :-1]) ** 2) ** 0.5

    # First derivative, central difference
    dx = (xy[0, 2:] - xy[0, :-2]) / (dt[1:] + dt[:-1])
    dy = (xy[1, 2:] - xy[1, :-2]) / (dt[1:] + dt[:-1])

    # Second derivative, central difference
    ddx = (xy[0, 2:] - 2 * xy[0, 1:-1] + xy[0, :-2]) / (dt[1:] * dt[:-1])
    ddy = (xy[1, 2:] - 2 * xy[1, 1:-1] + xy[1, :-2]) / (dt[1:] * dt[:-1])

    # Radius of curvature
    # https://en.wikipedia.org/wiki/Radius_of_curvature#In_two_dimensions
    k = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)

    # Pad values to maintain original array shape
    k = np.pad(k, (0, 2), mode='constant', constant_values=k[-1])

    return k