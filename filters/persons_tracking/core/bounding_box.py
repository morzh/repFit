from dataclasses import dataclass
from loguru import logger
from typing import Type

import numpy as np


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

    def circumscribe(self, bounding_box):
        if self._width == 0 and self._height == 0:
            self.__dict__ = bounding_box.__dict__
            return

        self._x = min(bounding_box.top_left[0], self._x)
        self._y = min(bounding_box.top_left[1], self._y)

        x2 = max(bounding_box.right_bottom[0], self.right_bottom[0])
        y2 = max(bounding_box.right_bottom[1], self.right_bottom[1])

        self.right_bottom = (x2, y2)
