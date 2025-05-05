# Computational Geometry
A collection of 2D computational geometry algorithms. Moslty pertaining to operations on 2D polylines.

## computational_geometry.py
Dependencies: Numpy
* *curvature_numeric(...)*: Find curvature of a polyline numerically
* *line_segment_y_value(...)*: Find the y value of a 2d line given x value and two endpoints
* *offset_cart(...)*: Generate polyline offset
* *offset_polar(...)*: Curve offsetting algorithm for polyline represented in polar coordinates.
* *polyline_boundary(...)*: Find upper/lower boundary (max/min y value) of polyline
* *polyline_circle_intersect(...)*: Find the intersections between an arbitrary polyline and a circle
* *polyline_self_intersect_naive(...)*: Find locations where a polyline intersects itself using brute force algorithm. Time complexity is O(n^2).
* *polyline_trim_loops(...)*: Find loops within a polyline and remove them. Uses naive intersection algorithm and therefore time complexity is O(n^2).

## test_cases.py
Dependencies: Numpy, Matplotlib, computational_geometry.py
Various tests of algorithms in computational_geometry.py.
