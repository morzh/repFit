from copy import deepcopy
import numpy as np

import numpy.typing as npt
from typing import Annotated, Literal,TypeVar

from geometry_typing import vec2d, line2d

class FloatComponent:
    """
    Description:
        Descriptor class for float component values.
    """
    __slots__ = ['name']

    def __set_name__(self, owner, name):
        self.name = '_' + name

    def __get__(self, instance, owner):
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        self.check_type(value)
        setattr(instance, self.name, value)

    @classmethod
    def check_type(cls, value):
        if not isinstance(value, float):
            raise ValueError('Value is not float')

class Line2D:
    """
    Description:
        Class with equation ax + by + c = 0, where (a, b, c) are so-called homogeneous line's coordinates.

    """
    # __slots__ = ['a', 'b', 'c']
    a = FloatComponent()
    b = FloatComponent()
    c = FloatComponent()
    def __init__(self, a, b, c,):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c
    
    def __repr__(self):
        return f"Line2D({self.a}, {self.b}, {self.c})"

    def __call__(self, *args) -> float:
        """
        Substitute point (x_0, y_0) to left hand side of line equation. In other words calculate a*x_0 + b*y_0 + c.

        :return: result of point substitution
        :raises: ValueError if argument is not tuple from two floats.
        """
        if isinstance(args, tuple) and len(args) == 1 and len(args[0]) == 2:  # TEST it !!!
            return self.a * args[0][0] + self.b * args[0][1] + self.c
        elif isinstance(args, np.ndarray) and args[0].shape[0] == 2:
            return self.a * args[0][0] + self.b * args[0][1] + self.c
        raise ValueError('Argument for __call__ should be single argument of type tuple[float, float]')

    @staticmethod
    def from_two_points(point_1: tuple[float, float] | np.ndarray, point_2: tuple[float, float] | np.ndarray) -> line2d:
        """
        Description:
            Constructs ``Line2D`` class  from two points.

        :param point_1: first point
        :param point_2: second point

        :return: ``Line2D`` class instance.
        """
        if isinstance(point_1, np.ndarray) and isinstance(point_2, np.ndarray):
            if point_1.shape[0] != 2 or point_2.shape[0] != 2:
                raise ValueError('point_1 and point_2 should be both 2D arrays')

        slope = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
        intercept = point_1[1] - slope * point_1[0]

        return Line2D.from_slope_and_intercept(slope, intercept)


    @staticmethod
    def from_point_and_direction(point: vec2d, direction: vec2d) -> line2d:
        r"""
        Description:
            Constructs ``Line2D`` class  from point and direction (2D vector).

            .. math::
                r(t) = t \cdot  direction + point = ⟨t \cdot k + x_0,t \cdot m + y_0⟩, t \in (-\infty, +\infty);

                t = \frac{x−x_0}{k} = \frac{y−y_0}{m};

                mx - ky  + (- m x_0 + k y_0) = 0.

        :param point: point, which belongs to the line.
        :param direction: line's direction.

        :return: ``Line2D`` class instance.
        """
        a = direction[1]
        b = -direction[0]
        c = -direction[1] * point[0] + direction[0] * point[1]
        return Line2D(a, b, c)

    @staticmethod
    def from_slope_and_intercept(k: float, b: float) -> line2d:
        """
        Description:
            The constant term b indicates the point where the line intersects the y-axis.
            The slope of the equation shows whether the line is ascending or descending.
            y = kx + b, where ``k`` is the slope and ``b`` -- constant term.

        :param k: line's slope
        :param b: line's intersect

        :return: Line2D class instance
        """
        return Line2D(k, -1, b)

    @staticmethod
    def from_intercepts(a: float, b: float, threshold: float = 1e-6) -> line2d:
        """
        Description:
            The intercept form of the equation of a line has an equation x/a + y/b = 1,
            where 'a' is the x-intercept, and 'b' is the y-intercept.

        :param a: x-intercept;
        :param b: y-intercept;
        :param threshold: parameters a and b absolute minimal value.

        :return: Line2D class instance or None (if line can not be constructed)
        """
        if abs(a) > threshold and abs(b) > threshold:
            return Line2D(1/a, 1/b, 1)  # TODO: think about extreme cases
        else:
            return Line2D(0, 0, 0)

    def intersection_point(self, other: line2d, threshold: float = 1e-6) -> vec2d | None:
        """
        Description:
            Calculates intersection point of this line and other line. Returns None is lines are parallel (within some threshold).

        :param other:
        :param threshold:

        :return: intersection point or None (if lines are parallel)
        """
        
        intersection_point_homogenous = np.cross(self.to_numpy(), other.to_numpy())
        
        if abs(intersection_point_homogenous[2]) < threshold:
            return None
        else:
            intersection_point_x = float(intersection_point_homogenous[0] / intersection_point_homogenous[2])
            intersection_point_y = float(intersection_point_homogenous[1] / intersection_point_homogenous[2])
            return np.array([intersection_point_x, intersection_point_y])


    def is_intersecting_with(self, other: line2d, threshold = 1e-6) -> bool:
        """
        Description:
        """
        this_normal = self.normal()
        other_normal = other.normal()

        this_normal /= np.linalg.norm(this_normal)
        other_normal /= np.linalg.norm(other_normal)

        projection = this_normal.dot(other_normal)

        if (1 - projection) < threshold:
            return False

        return True

    def distance_to_point(self, point: vec2d) -> float:
        """
        Description:
            Calculates closest distance from every point of this line to the given ``point``.
            For reference, see e.g.:  https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

        :param point: given point

        :return: distance to point
        """
        numerator = abs(self.a * point[0] + self.b * point[1] + self.c)
        denominator = np.sqrt(self.a ** 2 + self.b ** 2)
        return numerator / denominator

    def closest_point_on_line_to(self, point: vec2d) -> vec2d:
        """
        Description:
            Calculates the point on this line which is closest to the given ``point``.

        :param point: given point

        :return: closest point
        """
        denominator = np.sqrt(self.a ** 2 + self.b ** 2)
        closest_point_x = (self.b * (self.b * point[0]  - self.a * point[1]) - self.a * self.c) / denominator
        closest_point_y = (self.a * (-self.b * point[0] + self.a * point[1]) - self.b * self.c) / denominator

        return np.array([closest_point_x, closest_point_y])


    def distance_to_line(self, other_line: line2d) -> float:
        """
        Description:
            Calculates closest distance from every point of this line to every point of the  given ``line``.
            For reference, see e.g.: https://www.cuemath.com/geometry/distance-between-two-lines/

        :param other_line: given line

        :return: distance between lines

        """
        if self.is_intersecting_with(other_line):
            return 0.0

        numerator = abs(self.c - other_line.c)
        denominator = np.sqrt(self.a ** 2 + self.b ** 2)
        return numerator / denominator

    def normal(self) -> np.ndarray:
        """
        Description:
            Returns normal vector (un normalized)

        :return: normal vector
        """
        return np.array([-self.b, self.a])

    def is_parallel_to(self, other: line2d, threshold: float = 1e-6) -> bool:
        """
        Description:
            Checks if this line is parallel to other line.

        :param other: other line
        :param threshold: threshold for parallel lines check

        :return: True if lines are parallel (within some threshold), False otherwise.
        """
        intersection_point = self.intersection_point(other, threshold)  # TODO: get rid of the cross product in parallel lines detection

        if intersection_point is None:
            return True

        return False


    def copy(self) -> line2d:
        """
        Description:
            Return a deep copy of the object.
        """
        return deepcopy(self)
        
    def to_list(self) -> list:
        """
        Description:
            Return line coordinates as list.

        :return: line coordinates
        """
        return [self.a, self.b, self.c]
    
    def to_numpy(self) -> np.ndarray:
        """
        Description:
            Return line coordinates as numpy array.

        :return: line coordinates
        """
        return np.array([self.a, self.b, self.c]) 

