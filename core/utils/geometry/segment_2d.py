import enum
import numpy as np
from copy import deepcopy
from core.utils.geometry.line_2d import Line2D

from geometry_typing import vec2d, segment2d

class Segment2D:
    class Sign(enum.Enum):
        """
        Description
        """
        POSITIVE = 0
        NEGATIVE = 1

        def __neg__(self):
            if self.value == self.NEGATIVE:
                return self.POSITIVE
            else:
                return self.NEGATIVE

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

    def is_degenerate(self, threshold = 1e-6):
        """
        Description:
        """
        if self.length() < threshold:
            return True
        return False


    def is_inside_margins(self, point: vec2d, threshold: float = 1e-6) -> bool:
        """
        Description:
        """
        if self.length() < threshold and np.linalg.norm(self.start - point) < threshold:
            return True

        segment_normal = self.normal()
        margin_line_1 = Line2D.from_point_and_direction(self.start, segment_normal)
        margin_line_2 = Line2D.from_point_and_direction(self.end, segment_normal)


        check_point = 0.5 * (self.start + self.end)
        check_point_sign = self.__sign(margin_line_1(check_point))

        if self.__sign(margin_line_1(point)) == check_point_sign and self.__sign(margin_line_2(point)) == -check_point_sign:
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


    def distance(self, other: segment2d) -> tuple[float, vec2d]:
        """
        Description:
            Closest distance from this segment to the given one. THis method uses paper of
            Vladimir J. LUMELSKY. ON FAST COMPUTATION OF DISTANCE BETWEEN LINE SEGMENTS. 1984.
            https://jasoncantarella.com/octrope/trunk/doc/lumelsky.pdf

        :param other: given segment;

        :return: closest distance between segments.
        """

        if self.has_intersecting_with(other):
            return 0.0, self.intersection(other)

        distances = np.inf * np.ones(5)
        closest_points_candidates = np.inf * np.ones((8, 2))

        distances[0] = np.linalg.norm(self.start - other.start)
        distances[1] = np.linalg.norm(self.start - other.end)
        distances[2] = np.linalg.norm(self.end - other.start)
        distances[3] = np.linalg.norm(self.end - other.end)

        closest_points_candidates[0] = other.start
        closest_points_candidates[1] = other.end
        closest_points_candidates[2] = other.start
        closest_points_candidates[3] = other.end

        this_segment_line = Line2D.from_two_points(self.start, self.end)
        other_segment_line = Line2D.from_two_points(other.start, other.end)

        if self.is_parallel_to(other):
            distances[4] = this_segment_line.distance_to_line(other_segment_line)
            if self.is_inside_margins(other.start):
                closest_points_candidates[4] = other.start
            elif self.is_inside_margins(other.end):
                closest_points_candidates[4] = other.end
        else:
            closest_points_candidates[4] = this_segment_line.closest_point_on_line_to(self.start)

        current_closest_point = this_segment_line.closest_point_on_line_to(other.start)
        current_index = 4
        if self.is_inside_margins(current_closest_point):
            closest_points_candidates[current_index] = other.start
            distances[current_index] = np.linalg.norm(current_closest_point - other.start)
            current_index += 1

        current_closest_point = this_segment_line.closest_point_on_line_to(other.end)
        if self.is_inside_margins(current_closest_point):
            closest_points_candidates[current_index] = other.end
            distances[current_index] = np.linalg.norm(current_closest_point - other.end)
            current_index += 1

        current_closest_point = other_segment_line.closest_point_on_line_to(self.start)
        if other.is_inside_margins(current_closest_point):
            closest_points_candidates[current_index] = current_closest_point
            distances[current_index] = np.linalg.norm(current_closest_point - self.start)
            current_index += 1

        current_closest_point = other_segment_line.closest_point_on_line_to(self.end)
        if other.is_inside_margins(current_closest_point):
            closest_points_candidates[current_index] = current_closest_point
            distances[current_index] = np.linalg.norm(current_closest_point - self.end)

        minimum_distance_index = np.argmin(distances)
        return distances[minimum_distance_index], closest_points_candidates[minimum_distance_index]

    def normal(self, normalize=False) -> vec2d:
        """
        Description:
            Returns normal of the segment.

        :param normalize: If True normal vector will be normalized

        :return: normal to segment
        """
        direction_ = (self.end[0] - self.start[0], self.end[1] - self.start[1])
        normal_ = np.array([-direction_[1], direction_[0]])
        if normalize:
            return normal_ / np.linalg.norm(normal_)
        else:
            return normal_

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


    def length(self):
        return np.linalg.norm(self.end - self.start)

    def to_list(self):
        """
        Description:
        """
        return [[self.start[0], self.start[1]], [self.end[0], self.end[1]]]

    def to_tuples(self):
        """
        Description:
        """
        return (self.start[0], self.start[1]), ([self.end[0], self.end[1]])

    def copy(self) -> segment2d:
        """
        Description:
            Return a deep copy of the object.
        """
        return deepcopy(self)

    def __sign(self, value) -> Sign:
        """
        Description:
            Returns sign of a float value.

        :return: ``Sign`` enum value
        """
        sign_value = self.Sign.POSITIVE if value >= 0 else self.Sign.NEGATIVE
        return sign_value



