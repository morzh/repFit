from loguru import logger
from copy import deepcopy
from enum import Enum
import numpy as np


from bounding_box_mode import BoundingBoxMode
from geometry_typing import numeric, bbox2d, vec2d



class BoundingBox2D:
    """
    Description:
        BoundingBox2D class should serve for operations with bounding boxes.
    """

    class Order(Enum):
        VERTICAL = 0
        HORIZONTAL = 1

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['_x', '_y', '_width', '_height']
    def __init__(self, x: numeric = 0, y: numeric = 0, w_x2: numeric = 0, h_y2: numeric = 0, mode: BoundingBoxMode = XYWH):
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


    def __eq__(self, other: bbox2d) -> bool:
        if not isinstance(other, BoundingBox2D):
            return False
        return (self._x == other._x) and (self._y == other._y) and (self._width == other._width) and (self._height == other._height)

    def __repr__(self) -> str:
        return f"BoundingBox([{self._x}, {self._y}, {self._width}, {self._height}])"

    @staticmethod
    def from_list(values: list[float], mode: BoundingBoxMode = XYWH) -> bbox2d:
        """
        Description:
            Returns instance of the BoundingBox2D class from list of four values.

        :param values:
        :param mode:

        :return: BoundingBox2d instance
        """
        if len(values) == 4:
            if mode == BoundingBox2D.XYWH:
                return BoundingBox2D(values[0], values[1], values[2], values[3])
            elif mode == BoundingBox2D.XYXY:
                return BoundingBox2D(values[0], values[1], values[2] - values[0], values[3] - values[1])
        else:
            raise ValueError('')

    @staticmethod
    def from_numpy(values: np.ndarray, mode: BoundingBoxMode = XYWH) -> bbox2d:
        """
        Description:
            Returns instance of the BoundingBox2D class from list of four values.

        :param values:

        :param mode:

        :return: BoundingBox2d instance
        """
        if values.shape == (1, 4) or values.shape == (4,):
            if mode == BoundingBox2D.XYWH:
                return BoundingBox2D(values[0], values[1], values[2], values[3])
            elif mode == BoundingBox2D.XYXY:
                return BoundingBox2D(values[0], values[1], values[2] - values[0], values[3] - values[1])
        else:
            raise ValueError('')


    def apply_aspect_ratio(self, ratio: numeric) -> bbox2d:
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

    def to_list(self, mode:BoundingBoxMode=XYWH) -> list:
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

    def copy(self) -> bbox2d:
        """
        Description:
            Returns deep copy of this bounding box.

        :return: bounding box
        """
        return deepcopy(self)


    def is_degenerate(self, threshold=1e-9) -> bool:
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

    def contains_bounding_box(self, bounding_box: bbox2d, use_border=True) -> bool:
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

    def contained_in_bounding_box(self, other: bbox2d, use_border=True) -> bool:
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


    def shift(self, values: vec2d) -> bbox2d:
        """
        Description:
            Shifts bounding box by a given value.

        :param values: Shift value.

        :return: Shifted BoundingBox instance.
        """
        return BoundingBox2D(self._x + values[0], self._y + values[1], self._width, self._height)

    def scale(self, values: vec2d) -> bbox2d:
        """
        Description:
            Scales width and height. Top left corner remains the same.

        :param values: scale values

        :return: Scaled BoundingBox instance
        """
        return BoundingBox2D(self._x, self._y, self._width * values[0], self._height * values[1])

    def offset(self, value: numeric) -> bbox2d:
        """
        Description:
            Offsets each border segment of this bounding box by a certain value. Positive values decreases box area, negative increases.

        :param value: offset value

        :return: offset BoundingBox instance
        """
        if (2 * value > self._width) or (2 * value > self._height):
            return BoundingBox2D(0, 0, 0, 0)

        return BoundingBox2D(self._x + value, self._y + value, self._width - 2 * value, self._height - 2 * value)

    def enlarge(self, obstacles: list[bbox2d], bounding_box: bbox2d, order: Order.VERTICAL) -> bbox2d:
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

        if order == self.Order.VERTICAL:
            enlarged_box.__enlarge_vertically(obstacles_point_cloud, bounding_box)
            enlarged_box.__enlarge_horizontally(obstacles_point_cloud, bounding_box)
        elif order == self.Order.HORIZONTAL:
            enlarged_box.__enlarge_horizontally(obstacles_point_cloud, bounding_box)
            enlarged_box.__enlarge_vertically(obstacles_point_cloud, bounding_box)

        return enlarged_box


    def __enlarge_vertically(self, obstacles_point_cloud: np.ndarray, bounding_box: bbox2d) -> None:
        """
        Description:
        """
        points_above_top_segment = np.where(obstacles_point_cloud[:, 1] > self._y, obstacles_point_cloud)
        points_below_bottom_segment = np.where(obstacles_point_cloud[:, 1] > self._y + self._height, obstacles_point_cloud)

        minimal_distance_to_top_segment = 0
        if points_above_top_segment.size > 0:
            points_in_range_of_top_segment = points_above_top_segment[self._x < points_above_top_segment < self._x + self._width]
            minimal_distance_to_top_segment = np.minimum(points_in_range_of_top_segment - self._y)
            self._y -= minimal_distance_to_top_segment
        else:
            box_left_top = bounding_box.left_top
            self._y = box_left_top[1]
            self._height += self._y - box_left_top[1]

        if points_below_bottom_segment.size > 0:
            points_in_range_of_bottom_segment = points_below_bottom_segment[self._x < points_below_bottom_segment < self._x + self._width]
            minimal_distance_to_bottom_segment = np.minimum(points_in_range_of_bottom_segment - (self._y + self._height))
            self._height += minimal_distance_to_top_segment + minimal_distance_to_bottom_segment
        else:
            box_right_bottom = bounding_box.right_bottom
            self._height = box_right_bottom[1] - self._y

    def __enlarge_horizontally(self, obstacles_point_cloud: np.ndarray, bounding_box: bbox2d) -> None:
        """
        Description:
        """
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

        :param coordinates: new left top coordinates
        """
        self._x = coordinates[0]
        self._y = coordinates[1]

    @right_bottom.setter
    def right_bottom(self, coordinates: vec2d):
        """
        Description:
            Set bottom right coordinates of the bounding box.

        :param coordinates: new right bottom coordinates
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

        :param width: new width
        """
        self._width = abs(width)

    @height.setter
    def height(self, height: numeric) -> None:
        """
        Description:
            Sets height of the ``BoundingBox`` instance.

        :param height: new height
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
            Returns corners coordinates as [4, 2] numpy array. Order is left top, right top, right bottom, left bottm

        :return: corners coordinates array
        """
        return np.vstack((self.left_top, self.right_top, self.right_bottom, self.left_bottom))