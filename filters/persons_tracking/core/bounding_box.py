from typing import Type

import numpy as np
from dataclasses import dataclass
from loguru import logger
from typing import TypeVar
from typing import Union

numeric = Union[int, float]
BoundingBoxType = TypeVar("BoundingBoxType", bound="BoundingBox")

@dataclass
class BoundingBox:
    _x: numeric = -1
    _y: numeric = -1
    _width: numeric = 0
    _height: numeric = 0


    def is_degenerate(self) -> bool:
        """
        Description:
            Checks if bounding box is degenerate in other words has zero area.
        """
        return self.area == 0

    def has_bounding_box_inside(self, bounding_box: BoundingBoxType) -> bool:
        """
        Description:

        """

    def has_point_inside(self, point: tuple[numeric, numeric], use_closure=True ) -> bool:
        """
        Description:
            Checks if bounding box has given point inside of it. In case use_closure is True point could be at the border of the box.

        :return: True if point is inside bounding box, False otherwise.
        """
        if use_closure:
            return (self._x <= point[0] <= self._x + self._width) and (self._y <= point[1] <= self._y + self._height)
        else:
            return (self._x < point[0] < self._x + self._width) and (self._y < point[1] < self._y + self._height)


    def shift(self, value: tuple[numeric, numeric]) -> BoundingBoxType:
        """
        Description:
            Shifts bounding box by a given value.

        :return: Shifted BoundingBox instance
        """
        return BoundingBox(self._x + value[0], self._y + value[1], self._width, self._height)


    def scale_dimensions(self, value: tuple[numeric, numeric]) -> BoundingBoxType:
        """
        Description:
            Scales width and height. Top left corner remains the same.
        """
        return BoundingBox(self._x, self._y, self._width * value[0], self._height * value[1])

    def shrink_by(self, value: int):
        """
        Description:

        """

    def expand_by(self, value: int):
        """
        Description:

        """

    def expand_to(self, target: BoundingBoxType):
        """
        Description:

        """

    def intersect(self, bounding_box: BoundingBoxType):
        """
        Description:

        """

    def subtract(self, bounding_box: BoundingBoxType) -> list[BoundingBoxType]:
        """
        Description:

        """


    @property
    def top_left(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns top left coordinates of the bounding box.
        """
        return self._x, self._y

    @property
    def right_bottom(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns right bottom coordinates of the bounding box.
        """
        return self._x + self._width, self._y + self._height

    @property
    def width(self):
        """
        Description:
            Returns width of the bounding box.
        """
        return self._width

    @property
    def height(self):
        """
        Description:
            Returns height of the bounding box.
        """
        return self._height

    @property
    def area(self) -> numeric:
        """
        Description:
            Returns area of the bounding box.
        """
        return self._width * self._height

    @property
    def perimeter(self) -> numeric:
        """
        Description:
            Returns perimeter of the bounding box.
        """
        return self._width + self._height


    @top_left.setter
    def top_left(self, coordinates: tuple[numeric, numeric] | np.ndarray):
        """
        Description:

        """
        self._x = coordinates[0]
        self._y = coordinates[1]

    @right_bottom.setter
    def right_bottom(self, coordinates: tuple[numeric, numeric] | np.ndarray):
        """
            Description:

        """
        new_width = coordinates[0] - self._x
        new_height = coordinates[1] - self._y

        if new_width < 0 or new_height < 0:
            logger.warning('Right or bottom coordinate is incorrect')
            return

        if new_width == 0 or new_height == 0:
            logger.warning('Bounding box is degenerate')

        self._width = new_width
        self._height = new_height

    @width.setter
    def width(self, width: numeric) -> None:
        """
        Description:
            Sets width of the BoundingBox instance.
        """
        self._width = abs(int(width))

    @height.setter
    def height(self, height: numeric) -> None:
        """
        Description:
            Sets height of the BoundingBox instance.
        """
        self._height = abs(int(height))

    def circumscribe(self, bounding_box: BoundingBoxType) -> BoundingBoxType:
        """
        Description:

        """
        if self._width == 0 and self._height == 0:
            return bounding_box

        top_left = bounding_box.top_left
        right_bottom = bounding_box.right_bottom

        new_x = min(top_left[0], self._x)
        new_y = min(top_left[1], self._y)

        new_x2 = max(right_bottom[0], right_bottom[0])
        new_y2 = max(right_bottom[1], right_bottom[1])

        bounding_box_circumscribed = BoundingBox()
        bounding_box_circumscribed.top_left = (new_x, new_y)
        bounding_box_circumscribed.right_bottom = (new_x2, new_y2)

        return bounding_box_circumscribed
