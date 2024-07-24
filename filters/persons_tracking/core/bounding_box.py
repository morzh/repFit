from typing import Type

import numpy as np
from dataclasses import dataclass
from loguru import logger
from typing import TypeVar

BoundingBoxType = TypeVar("BoundingBoxType", bound="BoundingBox")


@dataclass
class BoundingBox:
    _x: int = -1
    _y: int = -1
    _width: int = 0
    _height: int = 0

    @property
    def top_left(self) -> tuple:
        return self._x, self._y

    @property
    def right_bottom(self):
        return self._x + self._width, self._y + self._height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def area(self):
        return self._width * self._height

    @top_left.setter
    def top_left(self, coordinates: tuple | list | np.ndarray):
        self._x = coordinates[0]
        self._y = coordinates[1]

    @right_bottom.setter
    def right_bottom(self, coordinates: tuple | list | np.ndarray):
        """
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
    def width(self, width):
        self._width = abs(int(width))

    @height.setter
    def height(self, height):
        self._height = abs(int(height))

    def circumscribe(self, bounding_box: BoundingBoxType) -> BoundingBoxType:
        if self._width == 0 and self._height == 0:
            return bounding_box

        new_x = min(bounding_box.top_left[0], self._x)
        new_y = min(bounding_box.top_left[1], self._y)

        new_x2 = max(bounding_box.right_bottom[0], self.right_bottom[0])
        new_y2 = max(bounding_box.right_bottom[1], self.right_bottom[1])

        bounding_box_circumscribed = BoundingBox()
        bounding_box_circumscribed.top_left = (new_x, new_y)
        bounding_box_circumscribed.right_bottom = (new_x2, new_y2)

        return bounding_box_circumscribed

    def intersect(self, bounding_box: BoundingBoxType):
        pass

    def subtract(self, bounding_box) -> list[BoundingBoxType]:
        pass

    def expand(self) -> BoundingBoxType:
        pass

    def shrink(self) -> BoundingBoxType:
        pass
