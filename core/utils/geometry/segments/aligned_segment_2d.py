from enum import Enum
import numpy as np
from copy import deepcopy
from core.utils.geometry.line_2d import Line2D
from core.utils.geometry.geometry_typing import vec2d, segment2d, alignedsegment2d, numeric


class AlignedSegmentType(Enum):
    """
    Description:
        AlignedSegment2D type enumeration.
    """
    VERTICAL = 0
    HORIZONTAL = 1


class AlignedSegment2D:
    """
    Description:
        Class, representing 2D segment, collinear with vector [1, 0] or [0, 1] .

    :ivar _x: segment start x component
    :ivar _y: segment start y component
    :ivar _length: segment length
    :ivar __type: segment type (horizontal or vertical)

    """
    def __init__(self, x, y, length, segment_type = AlignedSegmentType.VERTICAL):
        self._x = x
        self._y = y
        self._length = length
        self.__type = segment_type


    def is_less_than(self, threshold: numeric) -> bool:
        """
        Description:
            Checks if all segment's points are less than the ``threshold``.

        :param threshold: threshold value

        :return: True is condition satisfied, False otherwise.
        """
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x + self._length < threshold
        else:
            return self._y + self._length < threshold


    def is_less_or_equal_than(self, threshold: numeric) -> bool:
        """
        Description:
            Checks if all segment's points are less or equal than the ``threshold``.

        :param threshold: threshold value

        :return: True is condition satisfied, False otherwise.
        """
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x + self._length <= threshold
        else:
            return self._y + self._length <= threshold


    def is_greater_than(self, threshold: numeric) -> bool:
        """
        Description:
            Checks if all segment's points are greater than the ``threshold``.

        :param threshold: threshold value

        :return: True is condition satisfied, False otherwise.
        """
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x > threshold
        else:
            return self._y > threshold


    def is_greater_or_equal_than(self, threshold: numeric) -> bool:
        """
        Description:
            Checks if all segment's points are greater or equal than the ``threshold``.

        :param threshold: threshold value

        :return: True is condition satisfied, False otherwise.
        """
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x >= threshold
        else:
            return self._y >= threshold


    def endpoints_coordinates(self) -> tuple:
        """
        Description:
            Returns segment endpoints 2D coordinates.

        :return: segment's endpoints coordinates
        """
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return (self._x, self._y), (self._x + self._length, self._y)
        else:
            return (self._x, self._y), (self._x, self._y + self._length)


    @property
    def x(self) -> numeric:
        """
        Description:
            X component of segment start point getter.

        :return: x component value
        """
        return self._x


    @x.setter
    def x(self, value: numeric) -> None:
        """
        Description:
            X component of segment start point setter

        :param value: x component value
        """
        self._x = value


    @property
    def y(self) -> numeric:
        """
        Description:
            Y component of segment start point getter

        :return: y component value
        """
        return self._y


    @y.setter
    def y(self, value: numeric) -> None:
        """
        Description:
            Y component of segment start point setter

        :param value: y component value
        """
        self._y = value


    @property
    def length(self) -> numeric:
        """
        Description:
            Segment length getter

        :return: length
        """
        return self._length


    @length.setter
    def length(self, value: numeric) -> None:
        """
        Description:
            Segment length setter

        :param value:
        """
        self._length = abs(value)


    @property
    def type(self) -> AlignedSegmentType:
        """
        Description:
            Segment type getter
        """
        return self.__type


    @staticmethod
    def in_range(segments: list[alignedsegment2d], minimum, maximum, non_strict=True) -> list[alignedsegment2d]:
        """
        Description:
            Selects segments which are in ``minimum`` - ``maximum`` range.
            Segment considered as inclusive if at least oe point is with in given range.

        :param segments: list of input segments
        :param minimum: minimum bound
        :param maximum: maximum bound
        :param non_strict: if True, inequality is not strict. If False it is strict.

        :return: segments list
        """
        if non_strict:
            return [s for s in segments if not s.is_less_or_equal_than(minimum) and not s.is_greater_or_equal_than(maximum)]
        else:
            return [s for s in segments if not s.is_less_than(minimum) and not s.is_greater_than(maximum)]
