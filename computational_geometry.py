"""

"""

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# 2d Line segment y value
# ----------------------------------------------------------------------------------------------------------------------
def line_segment_y_value(x: float,
                         ep1: tuple[float, float] | list[float] | np.ndarray,
                         ep2: tuple[float, float] | list[float] | np.ndarray) -> tuple[float, float]:
    """
    Find the y value of a 2d line given x value and two endpoints
    :param x: value at which to find y
    :param ep1: x and y coordinates of an end point
    :param ep2: x and y coordinates of an end point
    :return: tuple containing the value of y, as well as curve parameter t, where t=0 is ep1 and t=1 is ep2
    """

    # Difference in end point x coordinates
    dx = (ep2[0] - ep1[0])

    # Flag indicating line is vertical
    is_vert = (dx == 0)

    # Calculate t
    if is_vert:
        # y-value is indeterminate; return t=1.0 as heuristic
        t = 1.0
    else:
        t = (x - ep1[0]) / dx

    # Calculate y
    y = ep1[1] + t * (ep2[1] - ep1[1])

    return y, t

# ----------------------------------------------------------------------------------------------------------------------
# Test Case
# ----------------------------------------------------------------------------------------------------------------------
def test_case():
    print(line_segment_y_value(1.0, (0, 0), (4, 4)))

# ---------------------------------------------------------------------------------------------------------------------
# Call to test case for validation purposes
# ---------------------------------------------------------------------------------------------------------------------
# Note: I think best practice is to save the test code in its own module, but I don't want to do that.
if __name__ == "__main__":
    test_case()