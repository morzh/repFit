import numpy as np
from bounding_box_2d import  BoundingBox2D
from geometry_typing import numeric, bbox2d


def intersect(self: bbox2d, other: bbox2d) -> bbox2d:
    """
    Description:
        Calculates intersection (which is also a box) of this bounding box with the ``target`` bounding box.
        If intersection is empty BoundingBox(0, 0, 0, 0) will be returned.

    :param self: first operand of intersect operation
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


def subtract(self: bbox2d, other: bbox2d) -> list[bbox2d]:
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

    :param self: first operand of subtract operation 
    :param other: bounding box to perform subtraction with

    :return: list of bounding boxes (result of subtraction).
    """

    if self.is_degenerate() or other.is_degenerate(): return list()

    intersected_bbox = self.intersect(other) # rect1 | rect2;
    if intersected_bbox.is_degenerate(): return list()

    intersections_grid = __intersections_grid(self, other)
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


def union(self: bbox2d, other: bbox2d) -> list[bbox2d]:
    """
    Description:
        Calculates bounding boxes union (which is a list of bounding boxes)

    :param self: first operand of union operation
    :param other: bounding box to perform union with

    :return: list of bounding boxes (result of union).
    """

    if self.is_degenerate() and other.is_degenerate(): return list()
    elif self.is_degenerate(): return [other]
    elif other.is_degenerate(): return [self]

    intersections_grid = __intersections_grid(self, other)
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


def circumscribe(self: bbox2d, bounding_box: bbox2d) -> bbox2d:
    """
    Description:
        Circumscribe this bounding box with the given one.

    :param self: first operand of circumscribe operation
    :param bounding_box: bounding box to circumscribe with

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

def intersection_over_union(self: bbox2d, bounding_box: bbox2d) -> numeric:
    """
    Description:
        Calculates intersection over union (IOU) metric.
        
    
    :param self: first operand of circumscribe operation
    :param bounding_box: bounding box to circumscribe with
    
    :return: IOU metric value
    """
    intersection_area = self.intersect(bounding_box).area
    union_area = self.area + bounding_box.area - self.intersect(bounding_box).area
    return intersection_area / union_area



def __intersections_grid(self: bbox2d, other: bbox2d) -> list[np.ndarray]:
    """
    Description:
        Calculates grid of points in the following way:
        1. Each border of this and other forms a line ( 4 horizontal and 4 vertical lines).
        2. grid of intersection each vertical line with horizontal line (total 16 points)

    :param other: bounding box to form intersection grid with this bounding box

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
