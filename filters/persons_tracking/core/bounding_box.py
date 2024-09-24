from typing import Type

from copy import deepcopy
import numpy as np
from dataclasses import dataclass
from loguru import logger
from typing import TypeVar
from typing import Union

from sympy import andre

numeric = Union[int, float]
BoundingBoxType = TypeVar("BoundingBoxType", bound="BoundingBox")

from enum import Enum


class BoxMode(Enum):
    """
    BoxMode2D defines the mode in which the bounding box is defined.

    Most data sources have bounding boxes defined as ``XYWH`` where `XY` is the top left corner \
        and `W` and `H` are the width and height of the box, respectively.

    However, many algorithms prefer to deal with bounding boxes as ``XYXY`` where the box is \
        defined is defined by the top-left corner and the bottom-right corner.

    To help disambiguate between these two configurations, `bbox` provides a means to specify the \
        mode and maintains the state internally.
    """
    XYWH = 0
    XYXY = 1


XYWH = BoxMode.XYWH.value
XYXY = BoxMode.XYXY.value


# @dataclass
class BoundingBox:
    """
    Description:
        BoundingBox class should serve for .
    """
    # _x: numeric = -1
    # _y: numeric = -1
    # _width: numeric = 0
    # _height: numeric = 0

    def __init__(self, x: numeric, y: numeric, w_x2: numeric, h_y2: numeric, mode: BoxMode = XYWH):
        if mode == XYWH:
            self._x = x
            self._y = y
            self._width = w_x2
            self._height = h_y2
        elif mode == XYXY:
            self._x = x
            self._y = y
            self._width = w_x2 - x
            self._height = h_y2 - y


    def __eq__(self, other: BoundingBoxType) -> bool:
        if not isinstance(other, BoundingBox):
            return False

        return (self._x == other._x) and (self._y == other._y) and (self._width == other._width) and (self._height == other._height)

    def __repr__(self):
        return f"BoundingBox([{self._x}, {self._y}, {self._width}, {self._height}])"

    def apply_aspect_ratio(self, ratio):
        """
        Description:
            Return bounding box mapped to new aspect ratio denoted by ``ratio``.

        Args:
            ratio (:py:class:`float`): The new ratio should be given as \
                the result of `width / height`.
        """
        # we need ratio as height/width for the below formula to be correct
        ratio = 1.0 / ratio

        area = self.w * self.h
        area_ratio = area / ratio
        new_width = np.round(np.sqrt(area_ratio))
        new_height = np.round(ratio * new_width)
        new_bbox = BoundingBox(self._x, self._y, new_width, new_height)
        return new_bbox

    def aspect_ratio(self):
        """
        Description:
        """
        return self._width / self._height

    def to_list(self, mode='xywh') -> list:
        """
        Description:
            Return bounding box as a `list` of 4 numbers. Format depends on ``mode`` flag (default is xywh).

        :param mode: Mode in which to return the box, 'xywh' ot 'xyxy'.

        :raise ValueError: if ``mode`` is not 'xywh' or 'xyxy'.

        :return: list of values
        """
        if mode == 'xywh':
            return [self._x, self._y, self._width, self._height]
        elif mode == 'xyxy':
            bottom_right = self.bottom_right
            return [self._x, self._y, bottom_right[0], bottom_right[1]]
        else:
            raise ValueError(f"Only 'xywh' and 'xyxy' modes are supported, but {mode} was given.")


    def to_numpy(self, mode='xywh') -> np.ndarray:
        """
        Description:
        """
        if mode == 'xywh':
            return np.array([self._x, self._y, self._width, self._height])
        elif mode == 'xyxy':
            bottom_right = self.bottom_right
            return np.array([self._x, self._y, bottom_right[0], bottom_right[1]])

    def copy(self):
        """
        Description:
            Return a deep copy of this bounding box.
        """
        return deepcopy(self)


    def is_degenerate(self, threshold=1e-6) -> bool:
        """
        Description:
            Checks if bounding box is degenerate in other words has zero area.

        :param threshold: Bounding box area threshold (for non integer numbers).

        :return: True if bounding box is degenerate, False otherwise.
        """
        return self.area <= numeric(threshold)

    def is_bounding_box_inside(self, bounding_box: BoundingBoxType) -> bool:
        """
        Description:
            Checks if this bounding box has another inside of it.

        :param bounding_box: Bounding box to check.

        :return: True if given bounding_box is inside, False otherwise.
        """

        return (self.contain_point(bounding_box.top_left) and
                self.contain_point(bounding_box.top_right) and
                self.contain_point(bounding_box.bottom_right) and
                self.contain_point(bounding_box.bottom_left))

    def is_bounding_box_outside(self, bounding_box: BoundingBoxType) -> bool:
        """
        Description:
            Checks if this bounding box has another outside of it.

        :param bounding_box: Bounding box to check.

        :return: True if given bounding_box is outside, False otherwise.
        """

    def contains_point(self, point: tuple[numeric, numeric], use_closure=True) -> bool:
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

    def offset(self, value: numeric) -> BoundingBoxType:
        """
        Description:
            Offsets each border segment of this bounding box by a certain value. Positive values decreases box area, negative increases.

        :param value:

        :return: offset BoundingBox instance
        """
        if (2 * value > self._width) or (2 * value > self._height):
            return BoundingBox(0, 0, 0, 0)

        return BoundingBox(self._x + value, self._y + value, self._width - 2 * value, self._height - 2 * value)

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
            Calculates intersection (which is also a box) of this bounding box with the given ``bounding_box``.
            If intersection is empty BoundingBox(0, 0, 0, 0) will be returned.

        :return: bounding box (result of intersection).
        """
        if self.is_degenerate() or bounding_box.is_degenerate(): return BoundingBox(0, 0, 0, 0)
        if self.is_bounding_box_outside(bounding_box):  return BoundingBox(0, 0, 0, 0)

        # /** projecting  horizontal side of the bounding_box angles to X axis */
        segments_x_begin = (self._x, bounding_box.x)
        segments_x_end = (self._x + self._width, bounding_box.x + bounding_box.width)
        # projecting vertical side of the rectangles to Y  axis
        segments_y_begin = (self._y, bounding_box.y)
        segments_y_end = (self._y + self._height, bounding_box.y + bounding_box.height)

        segments_x_intersection = (max(segments_x_begin[0], segments_x_begin[1]), min(segments_x_end[0], segments_x_end[1]))
        segments_y_intersection = (max(segments_y_begin[0], segments_y_begin[1]), min(segments_y_end[0], segments_y_end[1]))

        # /** check if rectangles have non empty intersection * /
        if (segments_x_intersection[1] < segments_x_intersection[0]) or (segments_y_intersection[1] < segments_y_intersection[0]):
            return BoundingBox(0, 0, 0, 0)

        # / ** do  inPlace assignment * /
        intersected_x = segments_x_intersection[0]
        intersected_y = segments_y_intersection[0]
        intersected_width = segments_x_intersection[1] - segments_x_intersection[0]
        intersected_height = segments_y_intersection[1] - segments_y_intersection[0]

        return  BoundingBox(intersected_x, intersected_y, intersected_width, intersected_height)


    def subtract(self, bounding_box: BoundingBoxType) -> list[BoundingBoxType]:
        r"""
        Description:
            Calculates subtraction (which is a list of bounding boxes) of this bounding box minus given ``bounding_box``.

                :math:`Rect_1 \setminus Rect_2`
            ┏━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃      Rect_1           ┃
            ┃                       ┃
            ┃    ┏━━━━━━━━━━━━━┓    ┃
            ┃    ┃ Rect_2      ┃    ┃
            ┃    ┗━━━━━━━━━━━━━┛    ┃
            ┃                       ┃
            ┃                       ┃
            ┗━━━━━━━━━━━━━━━━━━━━━━━┛

            If you subtract Rect_2 from Rect_1, you will get an area with a hole. This area can be decomposed into 4 rectangles
            ┏━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃          A            ┃
            ┃                       ┃
            ┣━━━━━┳━━━━━━━━━━━┳━━━━━┫
            ┃  B  ┃   hole    ┃  C  ┃
            ┣━━━━━┻━━━━━━━━━━━┻━━━━━┫
            ┃                       ┃
            ┃          D            ┃
            ┗━━━━━━━━━━━━━━━━━━━━━━━┛

        :return: list of bounding boxes (result of subtraction).
        """

        if self.is_degenerate(): return list()

        intersected_bbox = self.intersect(bounding_box) # rect1 | rect2;

        # Case 1. No intersection
        if intersected_bbox.is_degenerate(): return [self]

        raise NotImplementedError
        '''
        results = []
        remainder, subtractedArea

        subtractedArea = rectBetween(*this, intersected_bbox, & remainder, RectangleEdge::maxYEdge);
        if subtractedArea.area() != 0: results.append(subtractedArea);

        subtractedArea = rectBetween(remainder, intersected_bbox, & remainder, RectangleEdge::minYEdge);
        if subtractedArea.area != 0: results.append(subtractedArea);

        subtractedArea = rectBetween(remainder, intersected_bbox, & remainder, RectangleEdge::maxXEdge);
        if not subtractedArea.is_degenerate(): results.append(subtractedArea)

        subtractedArea = rectBetween(remainder, intersected_bbox, & remainder, RectangleEdge::minXEdge);
        if not subtractedArea.is_degenerate(): results.append(subtractedArea)
        return results
        '''

    def union(self, bounding_box: BoundingBoxType) -> list[BoundingBoxType]:
        """
        Description:
            Calculates bounding boxes union (which is a list of bounding boxes)

        :return: list of bounding boxes (result of union).
        """
        raise NotImplementedError


    def circumscribe(self, bounding_box: BoundingBoxType) -> BoundingBoxType:
        """
        Description:
            Circumscribe this bounding box with the given one.

        :param bounding_box:

        :return: bounding box
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


    def intersection_over_union(self, bounding_box: BoundingBoxType) -> numeric:
        """
        Description:
            Calculates intersection over union (IOU) metric.
        """
        intersection_area = self.intersect(bounding_box).area
        union_area = self.area + bounding_box.area - self.intersect(bounding_box).area
        return intersection_area / union_area

    def enlarge(self, obstacles: list[BoundingBoxType], bounding_box: BoundingBoxType) -> BoundingBoxType:
        """
        Description:

            1. All obstacle boxes (obstacles) are outside this box.
            2. This box is inside bounding_box
            3. For each line which goes along box border find minimal distance to obstacle boxes (-1 if not found).
            4. Offset each this box border by a value from step 3.


        :param obstacles:
        :param bounding_box:

        :return: enlarged bounding box

        """

        distances_to_neighbours = [self.distance_to(bbox) for bbox in obstacles]

        offset_value = min(distances_to_neighbours)
        offset_bbox = self.offset(offset_value)
        return offset_bbox.intersect(bounding_box)


    def distance_to(self, bounding_box: BoundingBoxType) -> numeric:
        """
        Description:
            Calculates shortest distance between this bounding box and given ``bounding_box``.

        :param bounding_box:

        :return: distance between bounding boxes
        """
        if self.intersect(bounding_box).is_degenerate():
            return -1

        raise NotImplementedError

    # def rectBetween(self):
    #     pass


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
    def width(self) -> numeric:
        """
        Description:
            Returns width of the bounding box.
            
        :return: bounding box width
        """
        return self._width

    @property
    def height(self) -> numeric:
        """
        Description:
            Returns height of the bounding box.
            
        :return: bounding box height
        """
        return self._height

    @property
    def area(self) -> numeric:
        """
        Description:
            Calculates area of the bounding box.
            
        :return: bounding box area
        """
        return self._width * self._height

    @property
    def perimeter(self) -> numeric:
        """
        Description:
            Calculates perimeter of the bounding box.
            
        :return: bounding box perimeter
        """
        return self._width + self._height


    @top_left.setter
    def top_left(self, coordinates: tuple[numeric, numeric] | np.ndarray):
        """
        Description:
            Set top left coordinates of the bounding box.

        """
        self._x = coordinates[0]
        self._y = coordinates[1]

    @bottom_right.setter
    def bottom_right(self, coordinates: tuple[numeric, numeric] | np.ndarray):
        """
        Description:
            Set bottom right coordinates of the bounding box.

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
    def width(self, width: numeric):
        """
        Description:
            Sets width of the ``BoundingBox`` instance.
        """
        self._width = abs(width)

    @height.setter
    def height(self, height: numeric) -> None:
        """
        Description:
            Sets height of the ``BoundingBox`` instance.
        """
        self._height = abs(height)

