from __future__ import annotations
from copy import deepcopy
from enum import Enum
from loguru import logger
import numpy as np

from core.utils.geometry.bounding_boxes.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.geometry_typing import numeric, vec2d
from core.utils.geometry.segments.aligned_segment_2d import AlignedSegment2D, AlignedSegmentType


VISUAL_DEBUG = False
if VISUAL_DEBUG:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import collections as mc

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
        LARGEST = 3

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


    def __eq__(self, other: BoundingBox2D) -> bool:
        if not isinstance(other, BoundingBox2D):
            return False
        return (self._x == other._x) and (self._y == other._y) and (self._width == other._width) and (self._height == other._height)


    def __repr__(self) -> str:
        return f"BoundingBox2D(x={self._x}, y={self._y}, width={self._width}, height={self._height})"


    @staticmethod
    def from_list(values: list[float], mode: BoundingBoxMode = XYWH) -> BoundingBox2D:
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
    def from_numpy(values: np.ndarray, mode: BoundingBoxMode = XYWH) -> BoundingBox2D:
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
    def from_two_points(point_1, point_2) -> BoundingBox2D:
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


    def copy(self) -> BoundingBox2D:
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
            return (np.all(self._x < points[:, 0]) and np.all(points[:, 0] < self._x + self._width) and
                    np.all(self._y < points[:, 1]) and np.all(points[:, 1] < self._y + self._height))


    def contains_bounding_box(self, bounding_box: BoundingBox2D, use_border=True) -> bool:
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


    def contained_in_bounding_box(self, other: BoundingBox2D, use_border=True) -> bool:
        """
        Description:
            Checks if this bounding box has ``other`` outside of it.

        :param other: Bounding box to check.
        :param use_border: Include bounding box border.

        :return: True if given bounding_box is outside, False otherwise.
        """
        points = self.corners()
        return other.contains_multiple_points(points, use_border=use_border)


    def shift(self, values: vec2d) -> BoundingBox2D:
        """
        Description:
            Shifts bounding box by a given value.

        :param values: Shift value.

        :return: Shifted BoundingBox instance.
        """
        return BoundingBox2D(self._x + values[0], self._y + values[1], self._width, self._height)


    def scale(self, values: vec2d) -> BoundingBox2D:
        """
        Description:
            Scales width and height. Top left corner remains the same.

        :param values: scale values

        :return: Scaled BoundingBox instance
        """
        return BoundingBox2D(self._x, self._y, self._width * values[0], self._height * values[1])


    def offset(self, value: numeric) -> BoundingBox2D:
        """
        Description:
            Offsets each border segment of this bounding box by a certain value. Positive values decreases box area, negative increases.

        :param value: offset value

        :return: offset BoundingBox instance
        """
        if value < 0 and abs(value) > self.minimum_dimension_value():
            return BoundingBox2D(0, 0, 0, 0)

        return BoundingBox2D(self._x - value, self._y - value, self._width + 2 * value, self._height + 2 * value)


    def extend(self, left: numeric | None = None, top: numeric | None = None, right: numeric | None = None, bottom: numeric | None = None) -> BoundingBox2D:
        """
        Description:
            Offset bounding box per side.

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
            box.height += bottom

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


    def enlarge(self, obstacle_bounding_boxes: list[BoundingBox2D], borderline_bounding_box: BoundingBox2D, order = Order.RANDOM) -> BoundingBox2D:
        """
        Description:
            Enlarges bounding box to the edges of given bounding boxes.

        :param obstacle_bounding_boxes: obstacles bounding boxes
        :param borderline_bounding_box: maximum bounding box. Enlarged bounding box will be inside this bounding box or will be degenerate
        :param order: order in which this bounding box will grow. It could be vertical, horizontal or random.

        :return: enlarged bounding box
        """
        enlarged_box = BoundingBox2D.intersect(self, borderline_bounding_box)
        if enlarged_box.is_degenerate():
            return enlarged_box

        if order == self.Order.RANDOM:
            order = np.random.randint(0, 2)

        if order == self.Order.VERTICAL:
            enlarged_box = enlarged_box.__enlarge_vertically(obstacle_bounding_boxes, borderline_bounding_box)
            enlarged_box = enlarged_box.__enlarge_horizontally(obstacle_bounding_boxes, borderline_bounding_box)
        elif order == self.Order.HORIZONTAL:
            enlarged_box = enlarged_box.__enlarge_horizontally(obstacle_bounding_boxes, borderline_bounding_box)
            enlarged_box = enlarged_box.__enlarge_vertically(obstacle_bounding_boxes, borderline_bounding_box)

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


    def __enlarge_horizontally(self, obstacle_boxes: list[BoundingBox2D], borderline_bounding_box: BoundingBox2D, numerical_tolerance=1e-6) -> BoundingBox2D:
        """
        Description:
            Enlarge this bounding box in horizontal direction.

        :param obstacle_boxes: inner obstacles bounding boxes
        :param borderline_bounding_box: outer obstacle bounding box

        :return: horizontally enlarged bounding box
        """
        left_side_segments = [box.right_segment() for box in obstacle_boxes]
        left_side_segments.append(borderline_bounding_box.left_segment())

        right_side_segments = [segment.left_segment() for segment in obstacle_boxes]
        right_side_segments.append(borderline_bounding_box.right_segment())

        left_side_segments = AlignedSegment2D.in_range(left_side_segments, self.left_top[1] + numerical_tolerance, self.left_bottom[1] - numerical_tolerance)
        left_side_segments = [segment for segment in left_side_segments if segment.x <= self._x + self._width]
        right_side_segments = AlignedSegment2D.in_range(right_side_segments, self.left_top[1] + numerical_tolerance, self.left_bottom[1] - numerical_tolerance)
        right_side_segments = [segment for segment in right_side_segments if segment.x >= self._x]

        left_side_segments_x = [segment.x for segment in left_side_segments]
        right_side_segments_x = [segment.x for segment in right_side_segments]

        left_side_segments_x_maximum = max(left_side_segments_x)
        right_side_segments_x_minimum = min(right_side_segments_x)

        enlarged_bounding_box = self.copy()
        if left_side_segments_x_maximum <= enlarged_bounding_box.x:
            enlarged_bounding_box = enlarged_bounding_box.extend(left=enlarged_bounding_box.x - left_side_segments_x_maximum)
        if right_side_segments_x_minimum >= enlarged_bounding_box.x + enlarged_bounding_box.width:
            enlarged_bounding_box = enlarged_bounding_box.extend(right=right_side_segments_x_minimum - enlarged_bounding_box.x - enlarged_bounding_box.width)

        if VISUAL_DEBUG:
            plot_left_side_segments = [s.endpoints_coordinates() for s in left_side_segments]
            plot_right_side_segments = [s.endpoints_coordinates() for s in right_side_segments]
            lines_left = mc.LineCollection(plot_left_side_segments, colors='red', linewidths=2)
            lines_right = mc.LineCollection(plot_right_side_segments, colors='blue', linewidths=2)
            fig, ax = plt.subplots()
            fig.set_size_inches(22.5, 14.5)
            plt.get_current_fig_manager().set_window_title('DEBUG Enlarge Horizontally')
            for box in obstacle_boxes:
                rect = Rectangle((box.x, box.y), width=box.width, height=box.height, edgecolor=(.5, .5, .5, .25), facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)
            ax.add_collection(lines_left)
            ax.add_collection(lines_right)

            rect = Rectangle((self.x, self.y), width=self.width, height=self.height,
                             edgecolor='green', facecolor=(1, 1, 1, 0), linewidth=3)
            ax.add_patch(rect)

            ax.add_patch(rect)
            rect = Rectangle((enlarged_bounding_box.x, enlarged_bounding_box.y), width=enlarged_bounding_box.width, height=enlarged_bounding_box.height,
                             edgecolor='green', facecolor=(1, 1, 1, 0))
            ax.add_patch(rect)
            plt.axvline(x=left_side_segments_x_maximum, color=(1, 0, 0, 0.25), label='axvline - full height')
            plt.axvline(x=right_side_segments_x_minimum, color=(0, 0, 1, 0.25), label='axvline - full height')
            ax.plot()
            plt.show()

        return enlarged_bounding_box


    def __enlarge_vertically(self, obstacle_boxes: list[BoundingBox2D], borderline_bounding_box: BoundingBox2D, numerical_tolerance=1e-6) -> BoundingBox2D:
        """
        Description:
            Enlarge this bounding box in vertical direction.

        :param obstacle_boxes: inner obstacles bounding boxes
        :param borderline_bounding_box: outer obstacle bounding box

        :return: vertically enlarged bounding box
        """
        top_side_segments = [box.bottom_segment() for box in obstacle_boxes]
        top_side_segments.append(borderline_bounding_box.top_segment())

        bottom_side_segments = [segment.top_segment() for segment in obstacle_boxes]
        bottom_side_segments.append(borderline_bounding_box.bottom_segment())

        top_side_segments = AlignedSegment2D.in_range(top_side_segments, self.left_top[0] + numerical_tolerance, self.right_top[0] - numerical_tolerance)
        top_side_segments = [segment for segment in top_side_segments if segment.y <= self._y + self._height]
        bottom_side_segments = AlignedSegment2D.in_range(bottom_side_segments, self.left_top[0] + numerical_tolerance, self.right_top[0] - numerical_tolerance)
        bottom_side_segments = [segment for segment in bottom_side_segments if segment.y >= self._y]

        top_side_segments_y = [segment.y for segment in top_side_segments]
        bottom_side_segments_y = [segment.y for segment in bottom_side_segments]

        top_side_segments_y_maximum = max(top_side_segments_y)
        bottom_side_segments_y_minimum = min(bottom_side_segments_y)

        enlarged_bounding_box = self.copy()
        if top_side_segments_y_maximum <= enlarged_bounding_box.y:
            enlarged_bounding_box = enlarged_bounding_box.extend(top=enlarged_bounding_box.y - top_side_segments_y_maximum)
        if bottom_side_segments_y_minimum >= enlarged_bounding_box.y + enlarged_bounding_box.height:
            enlarged_bounding_box = enlarged_bounding_box.extend(bottom=bottom_side_segments_y_minimum - enlarged_bounding_box.y - enlarged_bounding_box.height)


        if VISUAL_DEBUG:
            plot_top_side_segments = [s.endpoints_coordinates() for s in top_side_segments]
            plot_bottom_side_segments = [s.endpoints_coordinates() for s in bottom_side_segments]
            lines_top = mc.LineCollection(plot_top_side_segments, colors='red', linewidths=2)
            lines_bottom = mc.LineCollection(plot_bottom_side_segments, colors='blue', linewidths=2)

            fig, ax = plt.subplots()
            fig.set_size_inches(22.5, 14.5)
            plt.get_current_fig_manager().set_window_title('DEBUG Enlarge Vertically')

            for box in obstacle_boxes:
                rect = Rectangle((box.x, box.y), width=box.width, height=box.height, edgecolor=(.5, .5, .5, .25), facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)
            ax.add_collection(lines_top)
            ax.add_collection(lines_bottom)

            rect = Rectangle((self.x, self.y), width=self.width, height=self.height, edgecolor='green', facecolor=(1, 1, 1, 0), linewidth=3)
            ax.add_patch(rect)
            rect = Rectangle((enlarged_bounding_box.x, enlarged_bounding_box.y), width=enlarged_bounding_box.width, height=enlarged_bounding_box.height,
                             edgecolor='green', facecolor=(1, 1, 1, 0))
            ax.add_patch(rect)

            plt.axhline(y=top_side_segments_y_maximum, color=(1, 0, 0, 0.25), label='axhline - full width')
            plt.axhline(y=bottom_side_segments_y_minimum, color=(0, 0, 1, 0.25), label='axhline - full width')
            ax.plot()
            plt.show()

        return enlarged_bounding_box


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
    def right_bottom(self, coordinates: vec2d) -> None:
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
    
    
    @staticmethod
    def intersect(box_1: BoundingBox2D, box_2: BoundingBox2D)-> BoundingBox2D:
        """
        Description:
            Calculates intersection (which is also a box) of this bounding box with the ``target`` bounding box.
            If intersection is empty BoundingBox(0, 0, 0, 0) will be returned.

        :param box_1: first operand of intersect operation
        :param box_2: bounding box to perform intersection with.

        :return: bounding box (result of intersection).
        """
        if box_1.is_degenerate() or box_2.is_degenerate(): return BoundingBox2D(0, 0, 0, 0)
        elif box_1.contained_in_bounding_box(box_2): return box_1
        elif box_2.contained_in_bounding_box(box_1): return box_2

        # projecting  horizontal side of the bounding_box angles to X axis
        segments_x_begin = (box_1.x, box_2.x)
        segments_x_end = (box_1.x + box_1.width, box_2.x + box_2.width)
        # projecting vertical side of the rectangles to Y  axis
        segments_y_begin = (box_1.y, box_2.y)
        segments_y_end = (box_1.y + box_1.height, box_2.y + box_2.height)

        segments_x_intersection = (max(segments_x_begin[0], segments_x_begin[1]), min(segments_x_end[0], segments_x_end[1]))
        segments_y_intersection = (max(segments_y_begin[0], segments_y_begin[1]), min(segments_y_end[0], segments_y_end[1]))

        #  check if rectangles have non-empty intersection
        if (segments_x_intersection[1] < segments_x_intersection[0]) or (segments_y_intersection[1] < segments_y_intersection[0]):
            return BoundingBox2D(0, 0, 0, 0)

        intersected_x = segments_x_intersection[0]
        intersected_y = segments_y_intersection[0]
        intersected_width = segments_x_intersection[1] - segments_x_intersection[0]
        intersected_height = segments_y_intersection[1] - segments_y_intersection[0]

        return BoundingBox2D(intersected_x, intersected_y, intersected_width, intersected_height)


    @staticmethod
    def subtract(box_1: BoundingBox2D, box_2: BoundingBox2D) -> list[BoundingBox2D]:
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

        :param box_1: first operand of subtract operation
        :param box_2: bounding box to perform subtraction with

        :return: list of bounding boxes (result of subtraction).
        """

        if box_1.is_degenerate() or box_2.is_degenerate(): return list()

        intersected_bbox = BoundingBox2D.intersect(box_1, box_2)  # rect1 | rect2;
        if intersected_bbox.is_degenerate(): return list()

        intersections_grid = BoundingBox2D.__intersections_grid(box_1, box_2)
        subtraction_result = []

        for x_index in range(3):
            for y_index in range(3):
                current_left_top = intersections_grid[0][x_index], intersections_grid[1][y_index]
                current_width = intersections_grid[0][x_index + 1] - intersections_grid[0][x_index]
                current_height = intersections_grid[1][y_index + 1] - intersections_grid[1][y_index]
                current_bounding_box = BoundingBox2D(current_left_top[0], current_left_top[1], current_width, current_height)
                if box_1.contains_bounding_box(current_bounding_box) and not box_2.contained_in_bounding_box(current_bounding_box):
                    subtraction_result.append(current_bounding_box)

        raise subtraction_result


    @staticmethod
    def union(box_1: BoundingBox2D, box_2: BoundingBox2D) -> list[BoundingBox2D]:
        """
        Description:
            Calculates bounding boxes union (which is a list of bounding boxes)

        :param box_1: first operand of union operation
        :param box_2: bounding box to perform union with

        :return: list of bounding boxes (result of union).
        """

        if box_1.is_degenerate() and box_2.is_degenerate():
            return list()
        elif box_1.is_degenerate():
            return [box_2]
        elif box_2.is_degenerate():
            return [box_1]

        intersections_grid = BoundingBox2D.__intersections_grid(box_1, box_2)
        union_result = []

        for x_index in range(3):
            for y_index in range(3):
                current_left_top = intersections_grid[0][x_index], intersections_grid[1][y_index]
                current_width = intersections_grid[0][x_index + 1] - intersections_grid[0][x_index]
                current_height = intersections_grid[1][y_index + 1] - intersections_grid[1][y_index]
                current_bounding_box = BoundingBox2D(current_left_top[0], current_left_top[1], current_width, current_height)
                if box_1.contains_bounding_box(current_bounding_box) or box_2.contained_in_bounding_box(current_bounding_box):
                    union_result.append(current_bounding_box)

        raise union_result


    @staticmethod
    def circumscribe(bounding_box_1: BoundingBox2D, bounding_box_2: BoundingBox2D)-> BoundingBox2D:
        """
        Description:
            Circumscribe this bounding box with the given one.

        :param bounding_box_1: first operand of circumscribe operation
        :param bounding_box_2: bounding box to circumscribe with

        :return: bounding box
        """

        if bounding_box_1.width == 0 and bounding_box_1.height == 0:
            return bounding_box_2

        top_left = bounding_box_2.left_top
        right_bottom = bounding_box_2.right_bottom

        new_x = min(top_left[0], bounding_box_1.x)
        new_y = min(top_left[1], bounding_box_1.y)

        new_x2 = max(right_bottom[0], right_bottom[0])
        new_y2 = max(right_bottom[1], right_bottom[1])

        bounding_box_circumscribed = BoundingBox2D()
        bounding_box_circumscribed.left_top = (new_x, new_y)
        bounding_box_circumscribed.right_bottom = (new_x2, new_y2)

        return bounding_box_circumscribed


    @staticmethod
    def intersection_over_union(bounding_box_1: BoundingBox2D, bounding_box_2: BoundingBox2D) -> numeric:
        """
        Description:
            Calculates intersection over union (IOU) metric.


        :param bounding_box_1: first operand of circumscribe operation
        :param bounding_box_2: bounding box to circumscribe with

        :return: IOU metric value
        """
        intersection_area = BoundingBox2D.intersect(bounding_box_1, bounding_box_2).area()
        union_area = bounding_box_1.area() + bounding_box_2.area - BoundingBox2D.intersect(bounding_box_1, bounding_box_2).area()
        return intersection_area / union_area


    @staticmethod
    def __intersections_grid(bounding_box_1: BoundingBox2D, bounding_box_2: BoundingBox2D) -> list[np.ndarray]:
        """
        Description:
            Calculates grid of points in the following way:
            1. Each border of this and other forms a line ( 4 horizontal and 4 vertical lines).
            2. grid of intersection each vertical line with horizontal line (total 16 points)

        :param bounding_box_1: first bounding box to form intersection grid;
        :param bounding_box_2: second bounding box to form intersection grid.

        :return: points mesh grid
        """
        xs = np.zeros(4)
        ys = np.zeros(4)

        current_point = bounding_box_1.left_top
        xs[0] = current_point[0]
        ys[0] = current_point[1]

        current_point = bounding_box_1.right_bottom
        xs[1] = current_point[0]
        ys[1] = current_point[1]

        current_point = bounding_box_2.left_top
        xs[2] = current_point[0]
        ys[2] = current_point[1]

        current_point = bounding_box_2.right_bottom
        xs[3] = current_point[0]
        ys[3] = current_point[1]

        xs.sort()
        ys.sort()

        return np.meshgrid(xs, ys)
