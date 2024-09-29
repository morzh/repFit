from typing import TypeVar

from utils.geometry.line_2d import Line2D

vec2d = tuple[float, float]
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

    def __eq__(self, other: vec2d):
        return self.start == other.start and self.end == other.end

    def __repr__(self):
        return f"Segment2D({self.start}, {self.end})"

    def has_intersecting_with(self, other: segment2d) -> bool:
        line_1 = Line2D.from_two_points(self.start, self.end)
        line_2 = Line2D.from_two_points(other.start, other.end)

        if line_1(other.start) * line_1(other.end) < 0 and line_2(self.start) * line_2(self.end) < 0:
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

        :param other:

        :return: closest distance
        """

    def normal(self) -> tuple[float, float]:
        """
        Description:
        """
        direction = (self.end[0] - self.start[0], self.end[1] - self.start[1])
        return -direction[1], direction[0]
