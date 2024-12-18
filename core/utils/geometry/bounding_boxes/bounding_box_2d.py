from copy import deepcopy
from enum import Enum
from loguru import logger
import numpy as np

# import core.utils.geometry.bounding_boxes.bounding_box_2d_dyadic as bbox_dyadic
from core.utils.geometry.bounding_boxes.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.geometry_typing import numeric, bbox2d, vec2d
from core.utils.geometry.segments.aligned_segment_2d import AlignedSegment2D, AlignedSegmentType


class BoundingBox2D:
    """
    Description:
        Swiss army BoundingBox2D class.

    :ivar _x: top left x coordinate
    :ivar _y: top left y coordinate
    :ivar _width: bounding box width
    :ivar _height: bounding box height

    """

    class Order(Enum):
        VERTICAL = 0
        HORIZONTAL = 1
        RANDOM = 2

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
        return f"BoundingBox2D(x={self._x}, y={self._y}, width={self._width}, height={self._height})"


    @staticmethod
    def from_list(values: list[float], mode: BoundingBoxMode = XYWH) -> bbox2d:
        """
        Description:
            Returns instance of the BoundingBox2D class from list of four values.

        :param values:
        :param mode: BoundingBoxMode.XYWH or BoundingBoxMode.XYXY

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

        :param mode: BoundingBoxMode.XYWH or BoundingBoxMode.XYXY

        :return: BoundingBox2d instance
        """
        if values.shape == (1, 4) or values.shape == (4,):
            if mode == BoundingBox2D.XYWH:
                return BoundingBox2D(values[0], values[1], values[2], values[3])
            elif mode == BoundingBox2D.XYXY:
                return BoundingBox2D(values[0], values[1], values[2] - values[0], values[3] - values[1])
        else:
            raise ValueError('')


    @staticmethod
    def from_two_points(point_1, point_2) -> bbox2d:
        """
        Description:
            Returns instance of the BoundingBox2D class from list of four values.

        :param point_1: first bounding box corner
        :param point_2: second bounding box corner

        :return: BoundingBox2d instance

        :raises: ValueError if input arguments are not both numpy array or size is not 2
        """
        if not isinstance(point_1, np.ndarray) or not isinstance(point_2, np.ndarray):
            raise ValueError('Arguments both should be a numpy array')
        elif point_1.size != 2 or point_2.size != 2:
            raise ValueError('Arguments should be and numpy arrays of size 2')

        xyxy = np.vstack((point_1.reshape(1, 2), point_2.reshape(1, 2)))
        top_left = np.min(xyxy, axis=0)
        bottom_right = np.max(xyxy, axis=0)
        width_height = bottom_right - top_left
        return BoundingBox2D(top_left[0], top_left[1], width_height[0], width_height[1])


    @staticmethod
    def from_center_and_dimensions(center: vec2d, width: numeric, height: numeric):
        center = center.flatten()
        return BoundingBox2D(center[0] - 0.5 * width, center[1] - 0.5 * height, width, height)


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
        return self.area() < threshold


    def contains_single_point(self, point: vec2d, use_border=True) -> bool:
        """
        Description:
            Checks if bounding box has given point inside of it. In case use_closure is True point could be at the border of the box.

        :return: True if point is inside bounding box, False otherwise.

        :raises ValueError: If ``point`` is not of the proper size
        """
        if point.size != 2:
            raise ValueError('Input point size should be 2.')

        if use_border:
            return (self._x <= point[0] <= self._x + self._width) and (self._y <= point[1] <= self._y + self._height)
        else:
            return (self._x < point[0] < self._x + self._width) and (self._y < point[1] < self._y + self._height)


    def contains_multiple_points(self, points: np.ndarray, use_border=True):
        """
        Description:
            Checks if bounding box has given point inside of it. In case use_closure is True point could be at the border of the box.

        :return: True if point is inside bounding box, False otherwise.

        :raises ValueError: If ``points`` is not of the proper shape
        """
        if len(points.shape) != 2:
            raise ValueError('Input points shape should be Nx2.')

        if use_border:
            return (np.all(self._x <= points[:, 0]) and np.all(points[:, 0] <= self._x + self._width) and
                    np.all(self._y <= points[:, 1]) and np.all(points[:, 1] <= self._y + self._height))
        else:
            return (np.all(self._x < points[:, 0])  and np.all(points[:, 0] < self._x + self._width) and
                    np.all(self._y < points[:, 1])  and np.all(points[:, 1] < self._y + self._height))


    def contains_bounding_box(self, bounding_box: bbox2d, use_border=True) -> bool:
        """
        Description:
            Checks if this bounding box has another ``bounding_box`` inside of it.

        :param bounding_box: Bounding box to check.
        :param use_border: Include bounding box border

        :return: True if given bounding_box is inside, False otherwise.
        """
        return (self.contains_single_point(bounding_box.left_top, use_border) and
                self.contains_single_point(bounding_box.right_top, use_border) and
                self.contains_single_point(bounding_box.right_bottom, use_border) and
                self.contains_single_point(bounding_box.left_bottom, use_border))


    def contained_in_bounding_box(self, other: bbox2d, use_border=True) -> bool:
        """
        Description:
            Checks if this bounding box has ``other`` outside of it.

        :param other: Bounding box to check.
        :param use_border: Include bounding box border.

        :return: True if given bounding_box is outside, False otherwise.
        """
        points = self.corners()
        return other.contains_multiple_points(points, use_border=use_border)


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
        if value < 0 and abs(value) > self.minimum_dimension_value():
            return BoundingBox2D(0, 0, 0, 0)

        return BoundingBox2D(self._x - value, self._y - value, self._width + 2 * value, self._height + 2 * value)


    def extend(self, left: numeric | None = None, top: numeric | None = None, right: numeric | None = None, bottom: numeric | None = None) -> bbox2d:
        """
        Description:
            Offset bounding box in selected directions.

        :param left: if not None offsets left side of the bounding box
        :param top: if not None offsets top side of the bounding box
        :param right: if not None offsets right side of the bounding box
        :param bottom: if not None offsets bottom side of the bounding box

        :return: augmented bounding bbx
        """
        box = self.copy()
        if left is not None:
            box.x -= left
            box.width += left
        elif top is not None:
            box.y -= top
            box.height += top
        elif right is not None:
            box.width += right
        elif bottom is not None:
            box.width += bottom

        return box


    def left_segment(self) -> AlignedSegment2D:
        """
        Description:
            Returns left vertical segment of the bounding box.

        :return: vertical left segment
        """
        return AlignedSegment2D(self.left_top[0], self.left_top[1], self._height, AlignedSegmentType.VERTICAL)


    def right_segment(self) -> AlignedSegment2D:
        """
        Description:
            Returns right vertical segment of the bounding box.

        :return: vertical right segment
        """
        return AlignedSegment2D(self.right_top[0], self.right_top[1], self._height, AlignedSegmentType.VERTICAL)


    def top_segment(self) -> AlignedSegment2D:
        """
        Description:
            Returns left horizontal segment of the bounding box.

        :return: horizontal top segment
        """
        return AlignedSegment2D(self.left_top[0], self.left_top[1], self._width, AlignedSegmentType.HORIZONTAL)


    def bottom_segment(self) -> AlignedSegment2D:
        """
        Description:
            Returns left horizontal segment of the bounding box.

        :return: horizontal bottom segment
        """
        return AlignedSegment2D(self.left_bottom[0], self.left_bottom[1], self._width, AlignedSegmentType.HORIZONTAL)


    def enlarge(self, obstacle_bounding_boxes: list[bbox2d], borderline_bounding_box: bbox2d, order: Order.RANDOM) -> bbox2d:
        """
        Description:

            1. All obstacle boxes (obstacles) are outside this box.
            2. This box is inside ``borderline_bounding_box``

        :param obstacle_bounding_boxes:
        :param borderline_bounding_box:
        :param order: if VERTICAL enlargement first proceed in vertical direction, then in horizontal

        :return: enlarged bounding box
        """
        if not self.contained_in_bounding_box(borderline_bounding_box):
            return self

        obstacles_point_cloud = np.empty((0, 2))
        for obstacle in obstacle_bounding_boxes:
            obstacles_point_cloud = np.vstack((obstacles_point_cloud, obstacle.corners()))

        enlarged_box = self.copy()
        if order == self.Order.RANDOM:
            order = np.random.randint(0, 2)

        if order == self.Order.VERTICAL:
            enlarged_box.__enlarge_vertically(obstacle_bounding_boxes, borderline_bounding_box)
            enlarged_box.__enlarge_horizontally(obstacle_bounding_boxes, borderline_bounding_box)
        elif order == self.Order.HORIZONTAL:
            enlarged_box.__enlarge_horizontally(obstacles_point_cloud, borderline_bounding_box)
            enlarged_box.__enlarge_vertically(obstacles_point_cloud, borderline_bounding_box)

        return enlarged_box


    def maximum_dimension_value(self) -> numeric:
        """
        Description:
            Returns max(width, height)

        :return: maximum dimension value
        """
        return max(self._width, self._height)


    def minimum_dimension_value(self) -> numeric:
        """
        Description:
            Returns min(width, height)

        :return: minimum dimension value
        """
        return min(self._width, self._height)


    def center(self) -> vec2d:
        """
        Description:
            Gets center of the bounding box (half sum of left top corner right bottom corner).

        :return: center point
        """
        return np.array([self._x + 0.5 * self._width, self._y  + 0.5 * self._height])


    def corners(self) -> np.ndarray:
        """
        Description:
            Returns corners coordinates as [4, 2] numpy array. Order is left top, right top, right bottom, left bottom

        :return: vertices coordinates array
        """
        return np.vstack((self.left_top, self.right_top, self.right_bottom, self.left_bottom))


    def area(self) -> numeric:
        """
        Description:
            Calculates area of the bounding box.

        :return: bounding box area
        """
        return self._width * self._height


    def __enlarge_horizontally(self, obstacles: list[bbox2d], borderline_bounding_box: bbox2d) -> None:
        """
        Description:
            Enlarge this bounding box in horizontal direction.

        :param obstacles:
        :param borderline_bounding_box:
        """
        left_side_segments = [segment.right_segment() for segment in obstacles]
        left_side_segments.append(borderline_bounding_box.left_segment())

        right_side_segments = [segment.left_segment() for segment in obstacles]
        right_side_segments.append(borderline_bounding_box.right_segment())

        left_side_segments = AlignedSegment2D.out_of_range(left_side_segments, self.left_top[1], self.left_bottom[1])
        right_side_segments = AlignedSegment2D.out_of_range(right_side_segments, self.left_top[1], self.left_bottom[1])

        left_side_segments_x = [segment.x for segment in left_side_segments]
        right_side_segments_x = [segment.x for segment in right_side_segments]

        left_side_segments_x_maximum = max(left_side_segments_x)
        right_side_segments_x_minimum = min(right_side_segments_x)

        if left_side_segments_x_maximum < self._x:
            self.extend(left=self._x - left_side_segments_x_maximum)

        if right_side_segments_x_minimum > self._x + self._width:
            self.extend(right=right_side_segments_x_minimum - self._x - self._width)


    def __enlarge_vertically(self, obstacles: list[bbox2d], borderline_bounding_box: bbox2d) -> None:
        """
        Description:
            Enlarge this bounding box in vertical direction.

        :param obstacles:
        :param borderline_bounding_box:
        """
        top_side_segments = [segment.bottom_segment() for segment in obstacles]
        top_side_segments.append(borderline_bounding_box.top_segment())

        bottom_side_segments = [segment.top_segment() for segment in obstacles]
        bottom_side_segments.append(borderline_bounding_box.bottom_segment())

        top_side_segments = AlignedSegment2D.out_of_range(top_side_segments, self.left_top[0], self.right_top[0])
        bottom_side_segments = AlignedSegment2D.out_of_range(bottom_side_segments, self.left_top[0], self.right_top[0])

        top_side_segments_y = [segment.x for segment in top_side_segments]
        bottom_side_segments_y = [segment.x for segment in bottom_side_segments]

        top_side_segments_y_maximum = max(top_side_segments_y)
        bottom_side_segments_y_minimum = min(bottom_side_segments_y)

        if top_side_segments_y_maximum < self._y:
            self.extend(top=self._y - top_side_segments_y_maximum)

        if bottom_side_segments_y_minimum > self._y + self._height:
            self.extend(right=bottom_side_segments_y_minimum - self._y - self._height)


    def perimeter(self) -> numeric:
        """
        Description:
            Calculates perimeter of the bounding box.

        :return: bounding box perimeter
        """
        return 2 * (self._width + self._height)


    @property
    def x(self) -> numeric:
        return self._x


    @x.setter
    def x(self, value) -> None:
        self._x  = value


    @property
    def y(self) -> numeric:
        return self._y


    @y.setter
    def y(self, value) -> None:
        self._y = value


    @property
    def width(self) -> numeric:
        """
        Description:
            Returns width of the bounding box.

        :return: bounding box width
        """
        return self._width


    @width.setter
    def width(self, width: numeric):
        """
        Description:
            Sets width of the ``BoundingBox`` instance.

        :param width: new width
        """
        self._width = abs(width)


    @property
    def height(self) -> numeric:
        """
        Description:
            Returns height of the bounding box.

        :return: bounding box height
        """
        return self._height


    @height.setter
    def height(self, height: numeric) -> None:
        """
        Description:
            Sets height of the ``BoundingBox`` instance.

        :param height: new height
        """
        self._height = abs(height)


    @property
    def left_top(self) -> vec2d:
        """
        Description:
            Returns top left coordinates of the bounding box.
        """
        return np.array([self._x, self._y])


    @left_top.setter
    def left_top(self, coordinates: vec2d):
        """
        Description:
            Set top left coordinates of the bounding box.

        :param coordinates: new left top coordinates
        """
        self._x = coordinates[0]
        self._y = coordinates[1]


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


    @property
    def left_bottom(self) -> vec2d:
        """
        Description:
            Returns bottom left coordinates of the bounding box.
        """
        return np.array([self._x, self._y + self._height])
