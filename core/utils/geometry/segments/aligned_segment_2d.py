from enum import Enum
import numpy as np
from copy import deepcopy
from core.utils.geometry.line_2d import Line2D
from core.utils.geometry.geometry_typing import vec2d, segment2d, alignedsegment2d, numeric


class AlignedSegmentType(Enum):
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


    def is_less_than(self, value: numeric) -> bool:
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x + self._length < value
        else:
            return self._y + self._length < value


    def is_less_or_equal_than(self, value: numeric) -> bool:
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x + self._length <= value
        else:
            return self._y + self._length <= value


    def is_greater_than(self, value: numeric) -> bool:
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x > value
        else:
            return self._y > value


    def is_greater_or_equal_than(self, value: numeric) -> bool:
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return self._x >= value
        else:
            return self._y >= value


    def endpoints_coordinates(self) -> tuple:
        if self.__type == AlignedSegmentType.HORIZONTAL:
            return (self._x, self._y), (self._x + self._length, self._y)
        else:
            return (self._x, self._y), (self._x, self._y + self._length)


    @property
    def x(self) -> numeric:
        return self._x


    @x.setter
    def x(self, value: numeric) -> None:
        self._x = value


    @property
    def y(self) -> numeric:
        return self._y


    @y.setter
    def y(self, value: numeric) -> None:
        self._y = value


    @property
    def length(self) -> numeric:
        return self._length


    @length.setter
    def length(self, value: numeric) -> None:
        self._length = abs(value)


    @property
    def type(self) -> AlignedSegmentType:
        return self.__type


    @staticmethod
    def out_of_range(segments: list[alignedsegment2d], minimum, maximum, non_strict=True) -> list[alignedsegment2d]:
        if non_strict:
            return [s for s in segments if not s.is_less_or_equal_than(minimum) and not s.is_greater_or_equal_than(maximum)]
        else:
            return [s for s in segments if not s.is_less_than(minimum) and not s.is_greater_than(maximum)]
