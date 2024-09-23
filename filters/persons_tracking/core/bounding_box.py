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
    """
    Description:
        BoundingBox class should serve for .
    """
    _x: numeric = -1
    _y: numeric = -1
    _width: numeric = 0
    _height: numeric = 0


    def is_degenerate(self, threshold=1e-6) -> bool:
        """
        Description:
            Checks if bounding box is degenerate in other words has zero area.

        :param threshold: Bounding box area threshold (for non integer numbers).

        :return: True if bounding box is degenerate, False otherwise.
        """
        return self.area <= numeric(threshold)

    def has_bounding_box_inside(self, bounding_box: BoundingBoxType) -> bool:
        """
        Description:
            Checks if this bounding box has another inside of it.

        :param bounding_box: Bounding box to check.

        :return: True if given bounding_box is inside, False otherwise.

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


    def shift(self, values: tuple[numeric, numeric]) -> BoundingBoxType:
        """
        Description:
            Shifts bounding box by a given value.

        :param values: Shift value.

        :return: Shifted BoundingBox instance.
        """
        return BoundingBox(self._x + values[0], self._y + values[1], self._width, self._height)


    def scale_dimensions(self, values: tuple[numeric, numeric]) -> BoundingBoxType:
        """
        Description:
            Scales width and height. Top left corner remains the same.

        :param values: scale values

        :return: Scaled BoundingBox instance
        """
        return BoundingBox(self._x, self._y, self._width * values[0], self._height * values[1])

    def offset_by(self, value: numeric) -> BoundingBoxType:
        """
        Description:
            Offsets each border segment of this bounding box by a certain value.

        :param value:

        :return: offset BoundingBox instance
        """

    def offset_to(self, target: BoundingBoxType) -> BoundingBoxType:
        """
        Description:
            Offsets this bounding box in a way, when some border lies on the border of given bounding box.

        :param target:

        :return: offset BoundingBox instance
        """

    def intersect(self, bounding_box: BoundingBoxType) -> BoundingBoxType:
        """
        Description:
            Calculates intersection (which is also a box) of this bounding box with the given bounding_box.

        :return: bounding box (result of intersection).
        """

    def subtract(self, bounding_box: BoundingBoxType) -> list[BoundingBoxType]:
        """
        Description:
            Calculates subtraction (which is a list of bounding boxes) of this bounding box minus given bounding_box.

        :return: list of bounding boxes (result of subtraction).
        """


    def circumscribe(self, bounding_box: BoundingBoxType) -> BoundingBoxType:
        """
        Description:

        """
        if self._width == 0 and self._height == 0:
            return bounding_box

        top_left = bounding_box.top_left
        right_bottom = bounding_box.bottom_right

        new_x = min(top_left[0], self._x)
        new_y = min(top_left[1], self._y)

        new_x2 = max(right_bottom[0], right_bottom[0])
        new_y2 = max(right_bottom[1], right_bottom[1])

        bounding_box_circumscribed = BoundingBox()
        bounding_box_circumscribed.top_left = (new_x, new_y)
        bounding_box_circumscribed.bottom_right = (new_x2, new_y2)

        return bounding_box_circumscribed


    def intersection_over_union(self) -> None:
        """
        Description:
            Calculates intersection over union (IOU) metric.
        """

    def distance_to_bounding_box(self, bounding_box: BoundingBoxType) -> numeric:
        """
        Description:
            Calculates shortest distance between this bounding box and given bounding_box.

        :param bounding_box:

        :return: distance between bounding boxes
        """


    @property
    def top_left(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns top left coordinates of the bounding box.
        """
        return self._x, self._y


    @property
    def top_right(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns top right coordinates of the bounding box.
        """
        return self._x + self._width, self._y

    @property
    def bottom_right(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns bottom right coordinates of the bounding box.
        """
        return self._x + self._width, self._y + self._height

    @property
    def bottom_left(self) -> tuple[numeric, numeric]:
        """
        Description:
            Returns bottom left coordinates of the bounding box.
        """
        return self._x, self._y + self._height

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

    @bottom_right.setter
    def bottom_right(self, coordinates: tuple[numeric, numeric] | np.ndarray):
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

