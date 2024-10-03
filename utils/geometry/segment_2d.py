import numpy as np
import numpy.typing as npt
from utils.geometry.line_2d import Line2D

from typing import Annotated, Literal,TypeVar

vec2d = Annotated[npt.NDArray[np.float32 | np.float64], Literal[2]]
segment2d = TypeVar("segment2d", bound="Segment2D")

class Segment2D:
    """
    Description:
        Representing 2D segment using start and end points.
    """
    __slots__ = ['start', 'end']

    def __init__(self, start: vec2d, end: vec2d):
        self.start = start
        self.end = end

    def __eq__(self, other: segment2d, threshold = 1e-6):
        return (np.linalg.norm(self.start - other.start) < threshold and
                np.linalg.norm(self.end - other.end) < threshold)

    def __repr__(self):
        return f"Segment2D({self.start}, {self.end})"

    def has_intersecting_with(self, other: segment2d) -> bool:
        line_1 = Line2D.from_two_points(self.start, self.end)
        line_2 = Line2D.from_two_points(other.start, other.end)

        if line_1(other.start) * line_1(other.end) < 0 and line_2(self.start) * line_2(self.end) < 0:
            return True
        return False

    def is_parallel_to(self, other: segment2d, threshold = 1e-6) -> bool:
        """
        Description:
            Checks if segments are parallel

        :return: True if segments are parallel, False otherwise
        """
        direction_self = self.direction(normalize=True)
        direction_other = self.direction(normalize=True)

        projection = direction_self[0] * direction_self[0] +  direction_self[1] * direction_other[1]
        if (1 - projection) < threshold:
            return True

        return False


    def intersection(self, other: segment2d) -> vec2d | None:
        """
        Description:
            Calculates intersection point of this segment with the given onr.

        :param other:

        :return: intersection point (if any) or None (if there is no intersection)
        """
        if self.has_intersecting_with(other):
            line_1 = Line2D.from_two_points(self.start, self.end)
            line_2 = Line2D.from_two_points(other.start, other.end)
            return line_1.intersection_point(line_2)
        else:
            return None

    def distance(self, other: segment2d) -> float:
        """
        Description:
            Closest distance from this segment to the given one.
            https://stackoverflow.com/questions/54485106/finding-a-distance-between-two-line-segments

            Vladimir J. LUMELSKY. ON FAST COMPUTATION OF DISTANCE BETWEEN LINE SEGMENTS. 1984.
            https://jasoncantarella.com/octrope/trunk/doc/lumelsky.pdf

        :param other: given segment;

        :return: closest distance between segments.
        """

        if self.has_intersecting_with(other):
            return 0.0

        distances_endpoints = [None] * 4
        distances_projections = []




    def normal(self, normalize=False) -> vec2d:
        """
        Description:
        """
        direction = (self.end[0] - self.start[0], self.end[1] - self.start[1])
        normal = np.array([-direction[1], direction[0]])
        if normalize:
            return normal / np.linalg.norm(normal)
        else:
            return normal

    def direction(self, normalize = False) -> vec2d:
        """
        Description:
            Calculates direction of the segment. It is difference between end and start points.

        return: direction.
        """
        direction = self.end - self.start
        if normalize:
            return direction / np.linalg.norm(direction)
        else:
            return direction


    def to_list(self):
        """

        """
        return [[self.start[0], self.start[1]], [self.end[0], self.end[1]]]

    def to_numpy(self):
        """

        """
        return np.array([[self.start[0], self.start[1]], [self.end[0], self.end[1]]])