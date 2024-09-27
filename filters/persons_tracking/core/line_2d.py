from copy import deepcopy
import numpy as np

from typing import TypeVar

Line2DType = TypeVar("Line2DType", bound="Line2D")


class FloatComponent:
    """
    Description:
        Descriptor class for float component
    """
    __slots__ = ['value']

    def __init__(self, value=0):
        self.value = value

    def __set_name__(self, owner, name):
        self.value = '_' + name

    def __get__(self, instance, owner):
        return getattr(instance, 'value')

    def __set__(self, instance, value):
        self.check_type(value)
        setattr(instance, 'value', value)

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

    @staticmethod
    def from_two_points(point_1: tuple[float, float], point_2: tuple[float, float]) -> Line2DType:
        """
        Description:
            Constructs ``Line2D`` class  from two points.

        :param point_1: first point
        :param point_2: second point

        :return: ``Line2D`` class instance.
        """
        slope = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
        intercept = point_1[1] - slope * point_1[0]

        return Line2D.from_slope_intercept(slope, intercept)

    @staticmethod
    def from_point_and_direction(point: tuple[float, float], direction: tuple[float, float]) -> Line2DType:
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
    def from_slope_intercept(k: float, b: float) -> Line2DType:
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
    def from_intercepts(a: float, b: float, threshold: float = 1e-6) -> Line2DType:
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
            return Line2D(1/a, 1/b, 1)
        else:
            return Line2D(0, 0, 0)

    def intersection_point(self, other: Line2DType, threshold: float = 1e-6) -> tuple[float, float] | None:
        """
        Description:
            Calculates intersection point of this line and other line. Returns None is lines are parallel (within some threshold).

        :param other:
        :param threshold:

        :return: intersection point or None (if lines are parallel)
        """
        
        intersection_point = np.cross(self.to_numpy(), other.to_numpy())  # TODO: get rid of numpy cross product
        
        if abs(intersection_point[2]) < threshold:
            return None
        else:
            return float(intersection_point[0] / intersection_point[2]), float(intersection_point[1] / intersection_point[2])


    def distance_to_point(self, point: tuple[float, float]) -> float:
        """
        Description:
            Calculates closest distance from every point of this line to the given ``point``.

        :param point: given point

        :return: distance to point
        """
        return 0


    def distance_to_line(self, line: Line2DType) -> float:
        """
        Description:
            Calculates closest distance from every point of this line to every point of the  given ``line``.

        :param line: given line
        :return: distance between lines
        """
        return 0

    def normal(self) -> tuple[float, float]:
        """
        Description:
            Returns normal vector (un normalized)

        :return: normal vector
        """
        return -self.b, self.a

    def is_parallel_to(self, other: Line2DType, threshold: float = 1e-6) -> bool:
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


    def copy(self) -> Line2DType:
        """
        Description:
            Return a deep copy of object.
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

