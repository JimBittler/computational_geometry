import numpy as np
import pyclipper


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
def polyline_self_intersect_naive(xy: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Find locations where a polyline intersects itself using brute force algorithm. Time complexity is O(n^2).
    :param xy: a [2xn] array of cartesian coordinate pairs, [x, y], representing vertices of a polyline
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

def polyline_circle_intersect(xy: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    """
    Find the intersections between an arbitrary polyline and a circle
    :param xy: a [2xn] array of cartesian coordinate pairs, [x, y], representing vertices of a polyline
    :param c: coordinates of circle center, [x, y]
    :param r: circle radius
    :return: a [2xm] array of cartesian coordinate pairs, [x, y], representing intersections between the circle and polyline
    """
    # Get index of points which are within the bounding box of the circle, excluding curve endpoints
    this_idx = np.where(np.logical_and.reduce((
        xy[0, :] >= c[0] - r,
        xy[0, :] <= c[0] + r,
        xy[1, :] >= c[1] - r,
        xy[1, :] <= c[1] + r
    )))[0]

    # If no points in bounding box, return NaN
    if len(this_idx) == 0:
        return np.array(((np.nan,), (np.nan,)))

    # Get index of point which are within the circle
    this_idx = this_idx[((xy[0, this_idx] - c[0]) ** 2 + (xy[1, this_idx] - c[1]) ** 2) <= (r ** 2)]

    # If no points in circle, return NaN
    if len(this_idx) == 0:
        return np.array(((np.nan,), (np.nan,)))

    # Get index of segments which cross circle
    that_idx = np.where(np.diff(this_idx) != 1)[0]

    # Instantiate segment list
    s = []

    # Include first interior segment start and end point indices, unless it is the initial point of xy
    if this_idx[0] != 0:
        s = [(this_idx[0] - 1, this_idx[0])]

    # Get all other segment start and end point indices
    for ii in that_idx:
        s.append((this_idx[ii], this_idx[ii] + 1))
        s.append((this_idx[ii + 1] - 1, this_idx[ii + 1]))

    # Include final interior segment start and end point indices, unless it is the final point of xy
    if this_idx[-1] != xy.shape[1] - 1:
        s.append((this_idx[-1], this_idx[-1] + 1))

    # If no segments cross circle, return NaN
    if len(s) == 0:
        return np.array(((np.nan,), (np.nan,)))

    # VECTORIZE!
    ss = np.array(s).T

    # segment length along x and y axis
    dx = xy[0, ss[1, :]] - xy[0, ss[0, :]]
    dy = xy[1, ss[1, :]] - xy[1, ss[0, :]]

    # segment length squared
    rr = dx ** 2 + dy ** 2

    # Determinant of transformation matrix A
    # where A maps unit vectors with tails at circle center to
    # segment end points, with respect to circle center
    # A := [[s0x, s1x], [s0y, s1y]] - [[cx], [cy]]
    # |A| = ((s0x - cx) * (s1y - cy)) - ((s1x - cx) * (s0y - cy))
    dd = ((xy[0, ss[0, :]] - c[0]) * (xy[1, ss[1, :]] - c[1]) -
          (xy[0, ss[1, :]] - c[0]) * (xy[1, ss[0, :]] - c[1]))

    # Line circle intersection (-) w.r.t. circle center
    # https://mathworld.wolfram.com/Circle-LineIntersection.html
    xx = (dd * dy - (2 * (dy >= 0) - 1) * dx * np.sqrt(r ** 2 * rr - dd ** 2)) / rr
    yy = (-dd * dx - abs(dy) * np.sqrt(r ** 2 * rr - dd ** 2)) / rr

    # Check if the calculated intersection (-) is not contained within the line
    # parametric equation of a line:
    #   x = x0 + t * (xf - x0)  [1]
    #   y = y0 + t * (yf - y0)  [2]
    #   where t is in [0, 1]
    # If (xf - x0) == 0, i.e. a vertical line, solve for t using equation [2]
    # else solve for t using equation [1]

    # A small number
    epsilon = 1e-2  # ε = 0.001 → intersection is within a distance of 0.1% segment length from segment end points

    # Find dx ~= 0
    dx_is_zero = np.array((np.abs(dx) - epsilon < 0), dtype=bool)

    # Initialize boolean mask classifying if an intersection is within the line
    int_in_line = np.zeros_like(dx, dtype=bool)

    # Wherever dx == 0, use eq. [2] to solve for t
    # If t is in [0, 1], it is within the line segment
    # NOTE: Beware floating point error
    if dx_is_zero.any():
        int_in_line[dx_is_zero] = np.logical_and(
            (yy[dx_is_zero] + c[1] - xy[1, ss[0, dx_is_zero]]) / dy[dx_is_zero] >= -epsilon,
            (yy[dx_is_zero] + c[1] - xy[1, ss[0, dx_is_zero]]) / dy[dx_is_zero] <= 1 + epsilon
        )

    # Wherever dx != 0, use eq. [1] to solve for t
    # If t is in [0, 1], it is within the line segment
    if (~dx_is_zero).any():
        int_in_line[~dx_is_zero] = np.logical_and(
            (xx[~dx_is_zero] + c[0] - xy[0, ss[0, ~dx_is_zero]]) / dx[~dx_is_zero] >= -epsilon,
            (xx[~dx_is_zero] + c[0] - xy[0, ss[0, ~dx_is_zero]]) / dx[~dx_is_zero] <= 1 + epsilon
        )


    # If the intersection (-) is not within the segment, intersection (+) must be
    # Recall, a line segment will intersect with at circle 0, 1, or 2 real locations
    # All segments which do not intersect the center have already been eliminated
    # Recalculate appropriate intersections
    if (~int_in_line).any():
        xx[~int_in_line] = ((dd[~int_in_line] * dy[~int_in_line] +
                         (2 * (dy[~int_in_line] >= 0) - 1) * dx[~int_in_line] *
                         np.sqrt(r ** 2 * rr[~int_in_line] - dd[~int_in_line] ** 2)) / rr[~int_in_line])
        yy[~int_in_line] = ((-dd[~int_in_line] * dx[~int_in_line] +
                         abs(dy[~int_in_line]) *
                         np.sqrt(r ** 2 * rr[~int_in_line] - dd[~int_in_line] ** 2)) / rr[~int_in_line])

    # Shift intersections to global frame
    # Intersections were calculated w.r.t. circle center
    xx += c[0]
    yy += c[1]

    # assemble and return numpy array
    cc = np.array((xx, yy))

    return cc

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

    # Extrapolate values to pad start and end of line
    # --------------------------------
    tr = np.concatenate(
        ((tr[:, 0] - (tr[:, 1] - tr[:, 0]))[:, None],
         tr,
         (tr[:, -1] + (tr[:, -1] - tr[:, -2]))[:, None]),
        axis=1)

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

def offset_via_clipper(xy: np.ndarray = None, offset: float = 1.0, float2int_scale: int = 1000) -> tuple:
    """
    Offset a polygon via pyclipper library. Resultant offset curve is oriented counterclockwise, originating at the point which has a negative radial angle and is nearest the origin.
    Curve orientation has not been verified for robustness and may fail in the presences of degenerate inputs.
    :param xy: [2xn] Cartesian points of polygon vertices
    :param offset: distance to offset; negative "deflates", positive "inflates"
    :param float2int_scale: pyclipper only operates on integers. This value is used to scale floating point numbers. Effectively, this should represent 1e(n) where n is number is significant figures to the right of the decimal point in xy.
    :return: (xy, tr) tuple containing two [2xm] coordinates arrays Cartesian and polar coordinates. m not necessarily equal to n.
    """
    # pyclipper.PyclipperOffset() reference:
    # [1] https://angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/_Body.htm
    # [2] https://angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/_Body.htm

    # pyclipper can only operate on polygons with integer vertices
    # scale input and convert to integer
    xy *= float2int_scale

    # Transpose and convert to list to be comaptible with pyclipper
    xy = xy.astype('int').T.tolist()

    # subj = (xy.T).tolist()

    # Instantiate pyclipper offset object, which contains offset method
    pco = pyclipper.PyclipperOffset()

    # Add path to offset, with properties "Join Type square" and "End Type closed polygon"
    pco.AddPath(xy, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)

    # Calculate offset curve
    sol_xy = pco.Execute(int(offset * float2int_scale))[0]

    # Convert to [2xm] numpy array
    sol_xy = np.array(sol_xy, dtype=float).T

    # Convert to polar coordinates
    sol_tr = np.array(
        (
            np.arctan2(sol_xy[1, :], sol_xy[0, :]),
            np.hypot(sol_xy[1, :], sol_xy[0, :])
        )
    )

    # From clipper2 [2] docs, "Path order following offsetting very likely won't match path order prior to offsetting"
    # Reorder solution
    # 1) solution is composed of n points
    # 1) solution always has positive orientation; i.e. counterclockwise
    # 2) Find index point nearest origin
    # 3) beginning at nearest point found in (2), find first point which lies below the x-axis (negative angle)

    # smallest rho
    rho_min_idx = np.argmin(sol_tr[1, :])

    # negative phi index
    phi_mns_idx = np.argwhere(sol_tr[0, rho_min_idx:] < 0)

    # TODO: verify this
    # If negative phi not found after or including that of the minimum rho, check the solution array from the beginning
    if phi_mns_idx[0] == 0:
        phi_mns_idx = np.argwhere(sol_tr[0, :rho_min_idx] < 0)

        # if negative phi is not found, set the index to zero
        if phi_mns_idx[0] == 0:
            phi_mns_idx = 0
        # Else set the index to the first value
        else:
            phi_mns_idx = phi_mns_idx[0][0]
    else:
        # Else set the index to the first value
        phi_mns_idx = phi_mns_idx[0][0]

    # Reorder the solution
    sol_tr = np.concatenate(
        (
            sol_tr[:, (rho_min_idx + phi_mns_idx):],
            sol_tr[:, :(rho_min_idx + phi_mns_idx)]
        ),
        axis=1
    )

    # Cartesian coordinates
    sol_xy = sol_tr[1, :] * np.array((np.cos(sol_tr[0, :]), np.sin(sol_tr[0, :])))

    # Rescale
    sol_tr *= (1 / float2int_scale)
    sol_xy *= (1 / float2int_scale)

    return sol_xy, sol_tr

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
    insct_xy, insct_idx = polyline_self_intersect_naive(xy)

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

def polyline_boundary(xy: np.ndarray, upper: bool=True) -> np.ndarray:
    """
    Find upper/lower boundary (max/min y value) of polyline. Polyline may self intersect, but intersections are not
    detected and may degrade results. Theoretically O(n^2) complexity, but ~O(n) in practice. Theoretical O(nlog(n))
    complexity possible if segments are stored in sorted binary tree.
    :param xy: a [2xn] array of cartesian coordinate pairs representing vertices of a polyline
    :param upper: Flag if set to True, the upper boundary is found, else the lower boundary is found
    :return: a [2xm] array of  cartesian coordinate pairs representing vertices of a polyline boundary; m <= 2*(n-1)
    """

    # Set search index for calls to numpy.argpartition(...)
    # Upper: find the last element in a sorted array
    # else: find the first element in the sorted array
    if upper:
        kth = -1
    else:
        kth = 0

    # a small number
    # Note: this should be proportional to the input data
    epsilon = 1e-6

    # Dimension of polyline
    n_pts = xy.shape[1]

    # Line segment initial and end point indices
    # s = [s_{0}, s_{1}, ... , s_{n-1}] = [[s_{0,p0}, s_{0,pf}], [s_{1,p0}, s_{1,pf}], ... , [s_{n-1,p0}, s_{n-1,pf}]]
    # p0 := index of initial point in xy
    # pf := index of final point in xy
    # e.g. s[i] = [s_{i,p0}, s_{i,pf}]; ith line segment, s_{i}, which starts at point xy[:, s_{i,p0}] and ends at point xy[:, s_{i,pf}]
    # Note: initially, this is sequential because input is a polyline
    # Note: store in an array to make use of Numpy indexing routines
    # Note: structure could be exploited for optimization: i.e. adjacent segments always share one point
    s = np.array((np.arange(start=0, stop=n_pts - 1, step=1, dtype=int),
                  np.arange(start=1, stop=n_pts, step=1, dtype=int)), dtype=int)

    # change in x
    dx = np.diff(xy[0, :])

    # Check for vertical lines
    bool_idx = dx == 0
    if any(bool_idx):
        # Perturb vertical lines a small amount
        xy[0, 1:][bool_idx] += epsilon * np.random.rand(np.count_nonzero(bool_idx))

        # Recalculate change in x
        dx = np.diff(xy[0, :])

    # Identify "backwards" segments, where the final point is to the left of the initial point
    # Flip the initial point and end point indices of "backwards" segments
    bool_idx = dx < 0
    s[:, bool_idx] = np.flip(s[:, bool_idx], axis=0)

    # Get the indices that would sort the polyline by increasing x coordinate
    p_idx_sort = np.argsort(xy[0, :])

    # Initialize "status" T; a list of segments which intersect a horizontal sweepline
    # • The sweepline begins as the left most point
    # • Assumes there are no vertical lines
    # • Use a list because items will be added and removed frequently
    T = []

    # Add segments which start at the initial point to T
    # Note: this can be done faster segments are stored in a binary tree
    T.extend(np.where(s[0, :] == p_idx_sort[0])[0].tolist())

    # Set the active segment - segment which is the bound of the polyline at the sweepline location
    # • A = s[1, T] : Indices of end points of segments in T
    # • B = xy[1, A]: y values of end points
    # • C = np.argpartition(B, kth)[kth]: index of the kth largest value in B
    # • T[C]: segment in T which is the current boundary of the polyline
    s_act = T[np.argpartition(xy[1, s[1, T]], kth=kth)[kth]]

    # C: Points which are the bound of the polyline
    # • Initialize as the left most point
    # • Use a list because final size is unknown; upperbound is 2*(n-1)
    C = [(xy[0, p_idx_sort[0]], xy[1, p_idx_sort[0]])]

    # Advance the sweepline
    for this_idx in p_idx_sort[1:-1]:
        # # Useful debugging routine
        # print("====")
        # print(f"this_idx = {this_idx}")
        # print("status : ", T)
        # print(f"s_act: {s_act}")

        # Add next segment(s) to T
        # • can this be speed up by exploiting polyline structure? either the ith or (i-1)th segment will contain the ith point
        new_indices = np.where(s[0, :] == this_idx)[0].tolist()
        if len(new_indices) != 0:
            # Condition the new indices
            if upper:
                # Ensure the index of the top most new segment is placed at the end of the list of new segment indicies
                # This is done to ensure later calls to numpy.argpartition(...) pull the correct segment index
                new_indices.insert(len(new_indices), new_indices.pop(np.argpartition(xy[1, s[1, new_indices]], kth=kth)[kth]))
            else:
                # Ensure the index of the bottom most new segment is placed at the start of the list of new segment indicies
                # This is done to ensure later calls to numpy.argpartition(...) pull the correct segment index
                new_indices.insert(kth, new_indices.pop(np.argpartition(xy[1, s[1, new_indices]], kth=kth)[kth]))

            # Add new segment indices to status
            T.extend(new_indices)

        # Get max value
        # • x := sweepline position
        # • ep0 := initial points of segments in T
        # • ep1 := final points of segments in T
        this_y, _ = line_segment_y_value(x=xy[0, this_idx], ep0=xy[:, s[0, T]], ep1=xy[:, s[1, T]])

        # Get the index of kth y value of segments in T at x
        kth_idx = np.argpartition(this_y, kth=kth)[kth]

        # Add a new point to the output curve if either of the following is true:
        # • the new boundary point is not on the active segment
        # • the new boundary point is the endpoint of the active segment
        # Else no action required
        if T[kth_idx] != s_act or s[1, T[kth_idx]] == this_idx:
            # New boundary point is NOT on the active segment
            if T[kth_idx] != s_act:
                # Add a transition point: a point at the current x value and on the active segment
                C.append((xy[0, this_idx], this_y[T.index(s_act)]))
            # New boundary is the endpoint of the active segment
            else:
                # Add the end point
                C.append((xy[0, this_idx], this_y[kth_idx]))

            # Segments with endpoints on the sweepline should not be considered when finding a new active segment
            # Omit them by setting the y value of their endpoint(s) to +/- infinity
            # NOTE: There is a special case when the sweepline is at right most point of the polyline
            # upper == TRUE;  kth = -1 → -inf
            # upper == False; kth =  0 → +inf
            this_y[s[1, T] == this_idx] = (2 * kth + 1) * np.inf

            # find new active segment
            kth_idx = np.argpartition(this_y, kth=kth)[kth]

            # Add new point
            C.append((xy[0, this_idx], this_y[kth_idx]))

            # Update active segment
            s_act = T[kth_idx]

        # Remove segment from T if sweepline is at its endpoint
        T = [si for si in T if s[1, si] != this_idx]

    # Add final point
    this_idx = p_idx_sort[-1]
    new_indices = np.where(s[1, :] == this_idx)[0]
    this_y = np.partition(xy[1, new_indices], kth=kth)
    C.append((xy[0, this_idx], this_y[kth]))

    # convert to numpy array and return
    return np.array(C, dtype=float).T