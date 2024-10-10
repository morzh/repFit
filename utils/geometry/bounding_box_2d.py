from copy import deepcopy
from enum import Enum
from tkinter.constants import VERTICAL, HORIZONTAL

import numpy as np
from loguru import logger

import numpy.typing as npt
from typing import Annotated, Literal,TypeVar, Union

numeric = Union[int, float, np.float32, np.float64]
BoundingBox2DType = TypeVar("BoundingBox2DType", bound="BoundingBox2D")
vec2d = Annotated[npt.NDArray[np.float32 | np.float64], Literal[2]] | tuple[float, float]

class BoundingBox2D:
    """
    Description:
        BoundingBox class should serve for .
    """
    __slots__ = ['_x', '_y', '_width', '_height']
    class BoxMode(Enum):
        """
        Description:
            BoxMode defines the mode in which the bounding box is defined.

            Most data sources have bounding boxes defined as ``XYWH`` where `XY` is the top left corner
                and `W` and `H` are the width and height of the box, respectively.

            However, many algorithms prefer to deal with bounding boxes as ``XYXY`` where the box is \
                defined is defined by the top-left corner and the bottom-right corner.

            To help disambiguate between these two configurations, `bbox` provides a means to specify the \
                mode and maintains the state internally.
        """
        XYWH = 0
        XYXY = 1

    class Order(Enum):
        VERTICAL = 0
        HORIZONTAL = 1

    XYWH = BoxMode.XYWH.value
    XYXY = BoxMode.XYXY.value

    def __init__(self, x: numeric = 0, y: numeric = 0, w_x2: numeric = 0, h_y2: numeric = 0, mode: BoxMode = XYWH):
        if mode == BoundingBox2D.XYWH:
            self._x = x
            self._y = y
            self._width = w_x2
            self._height = h_y2
        elif mode == BoundingBox2D.XYXY:
            self._x = x
            self._y = y
            self._width = w_x2 - x
            self._height = h_y2 - y


    def __eq__(self, other: BoundingBox2DType) -> bool:
        if not isinstance(other, BoundingBox2D):
            return False
        return (self._x == other._x) and (self._y == other._y) and (self._width == other._width) and (self._height == other._height)

    def __repr__(self) -> str:
        return f"BoundingBox([{self._x}, {self._y}, {self._width}, {self._height}])"

    @staticmethod
    def from_list(values: list, mode: BoxMode = XYWH) -> BoundingBox2DType:
        pass

    @staticmethod
    def from_numpy(values: np.ndarray, mode: BoxMode = XYWH) -> BoundingBox2DType:
        pass

    def apply_aspect_ratio(self, ratio: numeric) -> BoundingBox2DType:
        """
        Description:
            Return bounding box mapped to new aspect ratio denoted by ``ratio``.

        :param ratio: The new ratio should be given as the result of `width / height`.
        :return: ``BoundingBox`` with new aspect ratio.
        """
        # we need ratio as height/width for the below formula to be correct
        ratio = 1.0 / ratio

        area = self._width * self._height
        area_ratio = area / ratio
        new_width = np.round(np.sqrt(area_ratio))
        new_height = np.round(ratio * new_width)
        new_bbox = BoundingBox2D(self._x, self._y, new_width, new_height)
        return new_bbox

    def aspect_ratio(self) -> numeric:
        """
        Description:
            Calculates aspect ratio, as width / height.

        :return: aspect ratio value
        """
        return self._width / self._height

    def to_list(self, mode:BoxMode=XYWH) -> list:
        """
        Description:
            Return bounding box as a `list` of 4 numbers. Format depends on ``mode`` flag (default is xywh).

        :param mode: output mode, either XYWH or XYXY.

        :return: list of values, representing bounding box

        :raise ValueError: If mode differs from XYWH or XYXY.
        """
        if mode == BoundingBox2D.XYWH:
            return [self._x, self._y, self._width, self._height]
        elif mode == BoundingBox2D.XYXY:
            bottom_right = self.right_bottom
            return [self._x, self._y, bottom_right[0], bottom_right[1]]
        else:
            raise ValueError('Modes other than XYWH and XYXY are not supported')

    def to_numpy(self, mode=XYWH) -> np.ndarray:
        """
        Description:
            Return [x, y, width, height] values as nump array. If ``mode``is XYWH.
            If ``mode`` is XYXY, returns [x_left, y_top, x_right, y_bottom] values as a numpy array.

        :param mode: output mode, either XYWH or XYXY.

        :return: bounding box numpy array

        :raise ValueError: If mode differs from XYWH or XYXY.
        """
        if mode == BoundingBox2D.XYWH:
            return np.array([self._x, self._y, self._width, self._height])
        elif mode == BoundingBox2D.XYXY:
            bottom_right = self.right_bottom
            return np.array([self._x, self._y, bottom_right[0], bottom_right[1]])
        else:
            raise ValueError('Modes other than XYWH and XYXY are not supported')

    def copy(self) -> BoundingBox2DType:
        """
        Description:
            Returns deep copy of this bounding box.

        :return: bounding box
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


    def contains_point(self, point: vec2d, use_border=True) -> bool:
        """
        Description:
            Checks if bounding box has given point inside of it. In case use_closure is True point could be at the border of the box.

        :return: True if point is inside bounding box, False otherwise.
        """
        if use_border:
            return (self._x <= point[0] <= self._x + self._width) and (self._y <= point[1] <= self._y + self._height)
        else:
            return (self._x < point[0] < self._x + self._width) and (self._y < point[1] < self._y + self._height)

    def contains_bounding_box(self, bounding_box: BoundingBox2DType, use_border=True) -> bool:
        """
        Description:
            Checks if this bounding box has another ``bounding_box`` inside of it.

        :param bounding_box: Bounding box to check.
        :param use_border: Include bounding box border

        :return: True if given bounding_box is inside, False otherwise.
        """

        return (self.contains_point(bounding_box.left_top, use_border) and
                self.contains_point(bounding_box.right_top, use_border) and
                self.contains_point(bounding_box.right_bottom, use_border) and
                self.contains_point(bounding_box.left_bottom, use_border))

    def contained_in_bounding_box(self, other: BoundingBox2DType, use_border=True) -> bool:
        """
        Description:
            Checks if this bounding box has ``other`` outside of it.

        :param other: Bounding box to check.
        :param use_border: Include bounding box border.

        :return: True if given bounding_box is outside, False otherwise.
        """
        if use_border:
            return self._x <= other.x and self._y <= other.y and self._width <= other.width and self._height <= other.height
        else:
            return self._x < other.x and self._y < other.y and self._width < other.width and self._height < other.height



    def shift(self, values: tuple[numeric, numeric]) -> BoundingBox2DType:
        """
        Description:
            Shifts bounding box by a given value.

        :param values: Shift value.

        :return: Shifted BoundingBox instance.
        """
        return BoundingBox2D(self._x + values[0], self._y + values[1], self._width, self._height)

    def scale(self, values: tuple[numeric, numeric]) -> BoundingBox2DType:
        """
        Description:
            Scales width and height. Top left corner remains the same.

        :param values: scale values

        :return: Scaled BoundingBox instance
        """
        return BoundingBox2D(self._x, self._y, self._width * values[0], self._height * values[1])

    def offset(self, value: numeric) -> BoundingBox2DType:
        """
        Description:
            Offsets each border segment of this bounding box by a certain value. Positive values decreases box area, negative increases.

        :param value: offset value

        :return: offset BoundingBox instance
        """
        if (2 * value > self._width) or (2 * value > self._height):
            return BoundingBox2D(0, 0, 0, 0)

        return BoundingBox2D(self._x + value, self._y + value, self._width - 2 * value, self._height - 2 * value)

    def intersect(self, other: BoundingBox2DType) -> BoundingBox2DType:
        """
        Description:
            Calculates intersection (which is also a box) of this bounding box with the ``target`` bounding box.
            If intersection is empty BoundingBox(0, 0, 0, 0) will be returned.

        :param other: bounding box to perform intersection with.

        :return: bounding box (result of intersection).
        """
        if self.is_degenerate() or other.is_degenerate(): return BoundingBox2D(0, 0, 0, 0)
        if self.contained_in_bounding_box(other):  return self

        # /** projecting  horizontal side of the bounding_box angles to X axis */
        segments_x_begin = (self._x, other.x)
        segments_x_end = (self._x + self._width, other.x + other.width)
        # projecting vertical side of the rectangles to Y  axis
        segments_y_begin = (self._y, other.y)
        segments_y_end = (self._y + self._height, other.y + other.height)

        segments_x_intersection = (max(segments_x_begin[0], segments_x_begin[1]), min(segments_x_end[0], segments_x_end[1]))
        segments_y_intersection = (max(segments_y_begin[0], segments_y_begin[1]), min(segments_y_end[0], segments_y_end[1]))

        # /** check if rectangles have non-empty intersection * /
        if (segments_x_intersection[1] < segments_x_intersection[0]) or (segments_y_intersection[1] < segments_y_intersection[0]):
            return BoundingBox2D(0, 0, 0, 0)

        # / ** do  inPlace assignment * /
        intersected_x = segments_x_intersection[0]
        intersected_y = segments_y_intersection[0]
        intersected_width = segments_x_intersection[1] - segments_x_intersection[0]
        intersected_height = segments_y_intersection[1] - segments_y_intersection[0]

        return  BoundingBox2D(intersected_x, intersected_y, intersected_width, intersected_height)


    def subtract(self, other: BoundingBox2DType) -> list[BoundingBox2DType]:
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

        :param other: bounding box to perform subtraction with

        :return: list of bounding boxes (result of subtraction).
        """

        if self.is_degenerate() or other.is_degenerate(): return list()

        intersected_bbox = self.intersect(other) # rect1 | rect2;
        if intersected_bbox.is_degenerate(): return list()

        intersections_grid = self.__intersections_grid(other)
        subtraction_result = []

        for x_index in range(3):
            for y_index in range(3):
                current_left_top = intersections_grid[0][x_index], intersections_grid[1][y_index]
                current_width = intersections_grid[0][x_index + 1] - intersections_grid[0][x_index]
                current_height = intersections_grid[1][y_index + 1] - intersections_grid[1][y_index]
                current_bounding_box = BoundingBox2D(current_left_top[0], current_left_top[1], current_width, current_height)
                if self.contains_bounding_box(current_bounding_box) and not other.contained_in_bounding_box(current_bounding_box):
                    subtraction_result.append(current_bounding_box)

        raise subtraction_result


    def union(self, other: BoundingBox2DType) -> list[BoundingBox2DType]:
        """
        Description:
            Calculates bounding boxes union (which is a list of bounding boxes)

        :param other: bounding box to perform union with

        :return: list of bounding boxes (result of union).
        """

        if self.is_degenerate() and other.is_degenerate(): return list()
        elif self.is_degenerate(): return [other]
        elif other.is_degenerate(): return [self]

        intersections_grid = self.__intersections_grid(other)
        union_result = []

        for x_index in range(3):
            for y_index in range(3):
                current_left_top = intersections_grid[0][x_index], intersections_grid[1][y_index]
                current_width = intersections_grid[0][x_index + 1] - intersections_grid[0][x_index]
                current_height = intersections_grid[1][y_index + 1] - intersections_grid[1][y_index]
                current_bounding_box = BoundingBox2D(current_left_top[0], current_left_top[1], current_width, current_height)
                if self.contains_bounding_box(current_bounding_box) or other.contained_in_bounding_box(current_bounding_box):
                    union_result.append(current_bounding_box)

        raise union_result


    def circumscribe(self, bounding_box: BoundingBox2DType) -> BoundingBox2DType:
        """
        Description:
            Circumscribe this bounding box with the given one.

        :param bounding_box:

        :return: bounding box
        """

        if self._width == 0 and self._height == 0:
            return bounding_box

        top_left = bounding_box.left_top
        right_bottom = bounding_box.right_bottom

        new_x = min(top_left[0], self._x)
        new_y = min(top_left[1], self._y)

        new_x2 = max(right_bottom[0], right_bottom[0])
        new_y2 = max(right_bottom[1], right_bottom[1])

        bounding_box_circumscribed = BoundingBox2D()
        bounding_box_circumscribed.left_top = (new_x, new_y)
        bounding_box_circumscribed.right_bottom = (new_x2, new_y2)

        return bounding_box_circumscribed

    def intersection_over_union(self, bounding_box: BoundingBox2DType) -> numeric:
        """
        Description:
            Calculates intersection over union (IOU) metric.
        """
        intersection_area = self.intersect(bounding_box).area
        union_area = self.area + bounding_box.area - self.intersect(bounding_box).area
        return intersection_area / union_area

    def enlarge(self, obstacles: list[BoundingBox2DType], bounding_box: BoundingBox2DType, order: Order.VERTICAL) -> BoundingBox2DType:
        """
        Description:

            1. All obstacle boxes (obstacles) are outside this box.
            2. This box is inside bounding_box
            3. For each line which goes along box border find minimal distance to obstacle boxes (-1 if not found).
            4. Offset each this box border by a value from step 3.

        :param obstacles:
        :param bounding_box:
        :param order: if VERTICAL enlargement first proceed in vertical direction, then in horizontal

        :return: enlarged bounding box
        """
        if not self.contained_in_bounding_box(bounding_box):
            return self

        obstacles_point_cloud = np.empty((0, 2))
        for obstacle in obstacles:
            obstacles_point_cloud = np.vstack((obstacles_point_cloud, obstacle.corners))


        enlarged_box = self.copy()

        if order == VERTICAL:
            enlarged_box.__enlarge_vertically(obstacles_point_cloud, bounding_box)
            enlarged_box.__enlarge_horizontally(obstacles_point_cloud, bounding_box)
        elif order == HORIZONTAL:
            enlarged_box.__enlarge_horizontally(obstacles_point_cloud, bounding_box)
            enlarged_box.__enlarge_vertically(obstacles_point_cloud, bounding_box)

        return BoundingBox2D()


    def __enlarge_vertically(self, obstacles_point_cloud: np.ndarray, bounding_box: BoundingBox2DType) -> None:
        points_above_top_segment = np.where(obstacles_point_cloud[:, 1] > self._y, obstacles_point_cloud)
        points_below_bottom_segment = np.where(obstacles_point_cloud[:, 1] > self._y + self._height, obstacles_point_cloud)

        minimal_distance_to_top_segment = 0
        if points_above_top_segmentsize.size > 0:
            points_in_range_of_top_segment = points_above_top_segment[self._x < points_above_top_segment < self._x + self._width]
            minimal_distance_to_top_segment = np.minimum(points_in_range_of_top_segment - self._y)
            self._y -= minimal_distance_to_top_segment
        else:
            box_left_top = bounding_box.left_top
            self._y = box_left_top[1]
            self._height += self._y - box_left_top[1]

        if points_in_range_of_bottom_segment.size > 0:
            points_in_range_of_bottom_segment = points_below_bottom_segment[self._x < points_below_bottom_segment < self._x + self._width]
            minimal_distance_to_bottom_segment = np.minimum(points_in_range_of_bottom_segment - (self._y + self._height))
            self._height += minimal_distance_to_top_segment + minimal_distance_to_bottom_segment
        else:
            box_right_bottom = bounding_box.right_bottom
            self._height = box_right_bottom[1] - self._y

    def __enlarge_horizontally(self, obstacles_point_cloud: np.ndarray, bounding_box: BoundingBox2DType) -> None:
        points_aside_left_segment = np.where(obstacles_point_cloud[:, 1] < self._x)
        points_aside_right_segment = np.where(obstacles_point_cloud[:, 1] > self._x + self._width)

        minimal_distance_to_left_segment = 0

        if points_aside_left_segment.size > 0:
            points_in_range_of_left_segment = points_aside_left_segment[self._y < points_aside_left_segment < self._y + self._height]
            minimal_distance_to_left_segment = np.minimum(points_in_range_of_left_segment - self._x)
            self._x -= minimal_distance_to_left_segment
        else:
            box_left_top = bounding_box.left_top
            self._x  = box_left_top[0]
            self._width += box_left_top[0] - self._x

        if points_aside_right_segment.size > 0:
            points_in_range_of_right_segment = points_aside_right_segment[self._y < points_aside_right_segment < self._y + self._height]
            minimal_distance_to_right_segment = np.minimum(points_in_range_of_right_segment - (self._x + self._width))
            self._width += minimal_distance_to_left_segment + minimal_distance_to_right_segment
        else:
            box_right_top = bounding_box.right_top
            self._width += box_left_top[0] - (self._y + self._width)


    @property
    def left_top(self) -> vec2d:
        """
        Description:
            Returns top left coordinates of the bounding box.
        """
        return np.array([self._x, self._y])

    @property
    def right_top(self) -> vec2d:
        """
        Description:
            Returns top right coordinates of the bounding box.
        """
        return np.array([self._x + self._width, self._y])

    @property
    def right_bottom(self) -> vec2d:
        """
        Description:
            Returns bottom right coordinates of the bounding box.
        """
        return np.array([self._x + self._width, self._y + self._height])

    @property
    def left_bottom(self) -> vec2d:
        """
        Description:
            Returns bottom left coordinates of the bounding box.
        """
        return np.array([self._x, self._y + self._height])

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


    @left_top.setter
    def left_top(self, coordinates: vec2d):
        """
        Description:
            Set top left coordinates of the bounding box.

        """
        self._x = coordinates[0]
        self._y = coordinates[1]

    @right_bottom.setter
    def right_bottom(self, coordinates: vec2d):
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

    @property
    def center(self) -> vec2d:
        """
        Description:
            Gets center of the bounding box (half sum of left top corner right bottom corner).

        :return: center point
        """
        return np.array([self._x + 0.5 * self._width, self._y  + 0.5 * self._height])

    @property
    def corners(self) -> np.ndarray:
        """
        Description:
        """
        return np.vstack((self.left_top, self.right_top, self.right_bottom, self.left_bottom))


    def __intersections_grid(self, other: BoundingBox2DType) -> list[np.ndarray]:
        """
        Description:
            Calculates grid of points in the following way:
            1. Each border of this and other forms a line ( 4 horizontal and 4 vertical lines).
            2. grid of intersection each vertical line with horizontal line (total 16 points)

        :param other: bounding box to form intersection grid

        :return: points mesh grid
        """
        xs = np.zeros(4)
        ys = np.zeros(4)

        current_point = self.left_top
        xs[0] = current_point[0]
        ys[0] = current_point[1]

        current_point = self.right_bottom
        xs[1] = current_point[0]
        ys[1] = current_point[1]


        current_point = other.left_top
        xs[2] = current_point[0]
        ys[2] = current_point[1]

        current_point = other.right_bottom
        xs[3] = current_point[0]
        ys[3] = current_point[1]

        xs.sort()
        ys.sort()

        return np.meshgrid(xs, ys)
