from __future__ import annotations
from enum import Enum
import numpy as np
import copy
import warnings

from core.utils.geometry.bounding_boxes.bounding_box_mode import BoundingBoxMode
from typing import Any


class BoundingBoxes2DArray:
    """
    Description:
        Class for storing and operating on bounding boxes array.
        Array of bounding boxes, represented by [top, left, width, height] format in image coordinates.

    Remarks:
        If some width or height equals to zero, this means the following bounding box degenerates to a segment.
        If both width and height equal zero, bounding box degenerates to a point.

    :ivar values: bounding boxes values (XYWH format).
    """

    class IntersectionMode(Enum):
        ONE_TO_MANY = 1
        ONE_TO_ONE = 2


    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['values']
    def __init__(self, bounding_boxes: np.ndarray | None = None, mode=XYWH):
        if bounding_boxes is not None:
            if mode == BoundingBoxes2DArray.XYXY:
                bounding_boxes = self.xyxy_to_xywh(bounding_boxes)
            elif not self.is_consistent(bounding_boxes):
                raise ValueError('Bounding boxes dimensions should be positive.')
            elif len(bounding_boxes.shape) != 2:
                raise ValueError('bounding_boxes should have two dimensions.')

            self.values = bounding_boxes
        else:
            self.values = np.empty((0, 4), dtype=np.int32)


    def __eq__(self, other):
        return np.array_equal(self.values, other.values)


    def __getitem__(self, item):
        return self.values[item]
    
    
    def __len__(self) -> int:
        return self.values.shape[0]


    def __iter__(self):
        for index in range(self.values.shape[0]):
            yield self.values[index]


    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        setattr(result, 'values', copy.deepcopy(self.values, memo))
        return result


    def reshape(self, new_shape):
        return self.values.reshape(new_shape)


    def append(self, bounding_box: np.ndarray, mode=XYWH) -> None:
        """
        Description:
            Append new bounding box to an existing array.

        :param bounding_box: bounding box to append;
        :param mode: bounding box format.
        """
        if bounding_box.size != 4:
            raise ValueError('Input bounding_box size should be 4')
        bounding_box_flatten = bounding_box.flatten()
        if (mode == BoundingBoxes2DArray.XYWH) and not np.alltrue(bounding_box_flatten[2:] > 0):
            raise ValueError('Input bounding_box[2:4] components should be greater or equal zero.')
        if mode == BoundingBoxes2DArray.XYXY:
            bounding_box = BoundingBoxes2DArray.xyxy_to_xywh(bounding_box)

        bounding_box = bounding_box.reshape((1, 4))
        self.values = np.vstack((self.values, bounding_box))


    def extend(self, bounding_boxes: np.ndarray, mode=XYWH) -> None:
        """
        Description:
            Add new bounding boxes to an existing array.

        :param bounding_boxes: bounding boxes to append (as numpy array);
        :param mode: bounding boxes format.
        """
        if bounding_boxes.shape[1] != 4 or len(bounding_boxes.shape) != 2:
            raise ValueError('Input bounding_box(es) should have size [N, 4].')
        elif (mode == BoundingBoxes2DArray.XYWH) and np.any(bounding_boxes[:, 2:] < 0):
            raise ValueError('Input bounding_boxes[:, 2:4] components should be greater or equal zero.')

        if mode == BoundingBoxes2DArray.XYXY:
            bounding_boxes = BoundingBoxes2DArray.xyxy_to_xywh(bounding_boxes)

        self.values = np.vstack((self.values, bounding_boxes))


    def circumscribe(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Circumscribe all bounding boxes. Result is also a bounding box.

        :param indices: indices of bounding boxes;

        :return: bounding box
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        top_lefts_minimum = np.min(selected_boxes[:, :2], axis=0)
        right_bottoms = selected_boxes[:, :2] + selected_boxes[:, 2:]
        right_bottoms_maximum = np.max(right_bottoms, axis=0)

        circumscribed_box = np.array([*top_lefts_minimum, *right_bottoms_maximum])
        return self.xyxy_to_xywh(circumscribed_box)


    def enlarge(self, top: np.ndarray | None = None, right: np.ndarray | None = None, bottom: np.ndarray | None = None, left: np.ndarray | None = None) -> BoundingBoxes2DArray:
        """
        Description:
            Enlarge bonding boxes in place by the given values.

        :param top:
        :param right:
        :param bottom:
        :param left:

        :return: enlarged bounding boxes array

        :raise ValueError: If argument(s) have / hase wrong shape or values
        """
        def check_enlarge_arguments(arg: np.ndarray):
            if not self.values.shape[0] ==  arg.size:
                raise ValueError('Wrong argument shape')
            if not np.all(arg >= 0):
                raise ValueError('All values argument should be non negative')

        enlarged_bounding_box = copy.deepcopy(self)

        if top is not None:
            check_enlarge_arguments(top)
            enlarged_bounding_box.values[:, 1] -= top
            enlarged_bounding_box.values[:, 3] += top
        if right is not None:
            check_enlarge_arguments(right)
            enlarged_bounding_box.values[:, 2] += right
        if bottom is not None:
            check_enlarge_arguments(bottom)
            enlarged_bounding_box.values[:, 3] += bottom
        if left is not None:
            check_enlarge_arguments(left)
            enlarged_bounding_box.values[:, 0] -= left
            enlarged_bounding_box.values[:, 2] += left

        return enlarged_bounding_box


    def areas(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Calculates areas of all bounding boxes.

        :param indices: indices of bounding boxes;

        :return: array of bounding box's areas.
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        return selected_boxes[:, 2] * selected_boxes[:, 3]


    def mean_area(self, indices: np.ndarray | None = None) -> np.floating[Any]:
        """
        Description:
            Mean area of bounding boxes with given ``indices``.

        :param indices: indices of bounding boxes. If None uses all bounding boxes in array for mean area calculation.

        :return: mean area of the (selected) bounding boxes
        """
        areas = self.areas(indices)
        return np.mean(areas)


    def mean_height(self, indices: np.ndarray | None = None) -> np.floating[Any]:
        """
        Description:
            Bounding boxes array heights Mean  with given ``indices``.

        :param indices: indices of bounding boxes. If None uses all bounding boxes in array for mean height calculation.

        :return: mean height of the (selected) bounding boxes
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        return np.mean(selected_boxes[:, 3])


    def perimeters(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Calculates perimeters of all bounding boxes.

        :param indices: indices of bounding boxes;

        :return: perimeters of the selected bounding boxes
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        return 2 * (selected_boxes[:, 2] + selected_boxes[:, 3])


    @property
    def shape(self) -> tuple:
        """
        Description:
            Bounding boxes array shape.

        :return: bounding boxes array shape
        """
        return self.values.shape


    @property
    def size(self) -> int:
        """
        Description:
            Bounding boxes array size.

        :return: bounding boxes array size
        """
        return self.values.size


    @staticmethod
    def is_consistent(bounding_boxes: np.ndarray) -> bool:
        """
        Description:
            Checks if 2nd and 3rd components of bounding boxes array are greater than zero.

        :return: True if consistent, False otherwise.
        """
        columns_number = bounding_boxes.shape[1]
        return np.alltrue(bounding_boxes[:, 2:] > 0) and columns_number == 4


    @staticmethod
    def intersect(boxes_1: BoundingBoxes2DArray, boxes_2: BoundingBoxes2DArray, mode = IntersectionMode.ONE_TO_MANY) -> list[BoundingBoxes2DArray] | BoundingBoxes2DArray:
        """
        Description:
            Intersects two bounding boxes arrays. In case both width and height equal zero,  no intersection occurred. This seems convenient for array processing.

        :param boxes_1: intersection first operand
        :param boxes_2: intersection second operand
        :param mode: intersection mode

        :return: ``boxes_1`` and ``boxes_2`` intersection result.

        :raise ValueError: If ``boxes_1`` and/or ``boxes_2`` has/have wrong shape.
        """
        if mode == BoundingBoxes2DArray.IntersectionMode.ONE_TO_MANY:
            return BoundingBoxes2DArray.__intersect_one_to_many(boxes_1, boxes_2)
        else:
            return BoundingBoxes2DArray.__intersect_one_to_one(boxes_1, boxes_2)


    @staticmethod
    def __intersect_one_to_one(boxes_1: BoundingBoxes2DArray, boxes_2: BoundingBoxes2DArray) -> BoundingBoxes2DArray:
        if not len(boxes_1) == len(boxes_2):
            raise ValueError('Number of bounding boxes for both operands should coincide.')

        boxes_1_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes_1.values)
        boxes_2_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes_2.values)

        left_tops = np.maximum(boxes_1_xyxy[:, :2], boxes_2_xyxy[:, :2])
        right_bottoms = np.minimum(boxes_1_xyxy[:, 2:], boxes_2_xyxy[:, 2:])
        intersection_xyxy = np.hstack((left_tops, right_bottoms))
        intersection_xywh = BoundingBoxes2DArray.xyxy_to_xywh(intersection_xyxy)

        return BoundingBoxes2DArray(intersection_xywh)


    @staticmethod
    def __intersect_one_to_many(boxes_1: BoundingBoxes2DArray, boxes_2: BoundingBoxes2DArray) -> list[BoundingBoxes2DArray]:
        if not len(boxes_1):
            return [BoundingBoxes2DArray()]
        elif not len(boxes_2):
            return [BoundingBoxes2DArray()] * len(boxes_1)

        boxes_1_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes_1.values)
        boxes_2_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes_2.values)

        intersected_boxes = [BoundingBoxes2DArray] * len(boxes_1)
        for box_1_index in range(boxes_1_xyxy.shape[0]):
            current_box_1_xyxy = boxes_1_xyxy[box_1_index]
            current_left_tops = np.vstack((current_box_1_xyxy[:2], boxes_2_xyxy[:, :2]))
            current_right_bottoms = np.vstack((current_box_1_xyxy[2:], boxes_2_xyxy[:, 2:]))
            current_left_top = np.min(current_left_tops, axis=0)
            current_right_bottom = np.max(current_right_bottoms, axis=0)
            current_xyxy = np.array([*current_left_top, *current_right_bottom])
            current_xywh = BoundingBoxes2DArray.xyxy_to_xywh(current_xyxy)
            intersected_boxes[box_1_index] = BoundingBoxes2DArray(current_xywh)

            return intersected_boxes


    @staticmethod
    def intersection_over_union(boxes_1: BoundingBoxes2DArray, boxes_2: BoundingBoxes2DArray) -> np.ndarray:
        """
        Description:
            Intersection over union measure for two arrays of bounding boxes

        :return: IoU values

        :raise ValueError: If ``boxes_1`` and ``boxes_2`` have different number of bounding boxes
        """
        if not len(boxes_1) == len(boxes_2):
            raise ValueError('Input arguments should be of the same size')

        union_areas = boxes_1.areas() + boxes_2.areas()
        intersections = BoundingBoxes2DArray.intersect(boxes_1, boxes_2, mode = BoundingBoxes2DArray.IntersectionMode.ONE_TO_ONE)
        intersection_areas = intersections.areas()
        return  union_areas / intersection_areas


    @staticmethod
    def clamp(boxes, clamp_box) -> BoundingBoxes2DArray:
        """
        Description:
            all ``boxes`` are inside ``clamp_box``.

        :param boxes: bounding boxes to clamp
        :param clamp_box: bounding box

        :return: clamped bounding boxes.

        :raises ValueError: If input arguments shape or size is incorrect.
        """

        if not (len(boxes.shape) == 2 and boxes.shape[1] == 4) or not (clamp_box.size == 4):
            raise ValueError('Boxes argument should have (N, 4) shape and clamp_box should have size of 4.')

        clamp_box = clamp_box.reshape(1, 4)
        boxes_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes)
        clamp_box_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(clamp_box).astype(np.int64)

        number_boxes = len(boxes)
        clamp_box_xyxy = np.repeat(clamp_box_xyxy, number_boxes, axis=0)
        top_lefts_clamped = np.clip(boxes_xyxy[:, :2], clamp_box_xyxy[:, :2], clamp_box_xyxy[:, 2:])
        bottom_rights_clamped = np.clip(boxes_xyxy[:, 2:], clamp_box_xyxy[:, :2], clamp_box_xyxy[:, 2:])

        clamped_boxes_xyxy = np.hstack((top_lefts_clamped, bottom_rights_clamped))

        # if not (np.alltrue(clamped_boxes_xyxy[:, :2] >= clamp_box_xyxy[:, :2])):
        #     print('!!!')

        clamped_boxes_xywh = BoundingBoxes2DArray.xyxy_to_xywh(clamped_boxes_xyxy)
        return BoundingBoxes2DArray(clamped_boxes_xywh)


    def __selected_bounding_boxes(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:

        :param indices:

        :return: selected by indices bounding boxes array.

        :raises ValueError:
        """
        if indices is None:
            return self.values

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Indices should be an array of integers.')
        elif len(indices.shape) != 1:
            raise ValueError('Indices should be an 1D array.')

        if indices.size > self.values.shape[0]:
            warnings.warn('Indices size bigger than the number of bounding boxes.')
            boxes_number = self.values.shape[0]
            indices = indices[:boxes_number]

        indices = np.sort(indices)
        if indices[-1] > (self.values.shape[0] - 1):
            raise ValueError('Indices are not correct.')

        return self.values[indices]


    @staticmethod
    def xyxy_to_xywh(bboxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.

        :param bboxes_xyxy: bounding boxes numpy array in XYXY format.

        :return: bounding boxes numpy array in XYWH format.
        """
        # TODO: think about degenerate cases
        xyxy = bboxes_xyxy.reshape((-1, 2, 2))
        xy_top_left = np.min(xyxy, axis=1)
        xy_bottom_right = np.max(xyxy, axis=1)
        xywh_boxes = np.hstack((xy_top_left, xy_bottom_right - xy_top_left))
        return xywh_boxes.reshape((-1, 4))

    @staticmethod
    def xywh_to_xyxy(bboxes_xywh: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.

        :param bboxes_xywh: bounding boxes numpy array in XYWH format.

        :return: bounding boxes numpy array in XYXY format.
        """
        if bboxes_xywh.size == 4:
            bboxes_xywh = bboxes_xywh.reshape((1, 4))

        xyxy_boxes =  np.hstack((bboxes_xywh[:, 0].reshape(-1, 1),
                                 bboxes_xywh[:, 1].reshape(-1, 1),
                                 (bboxes_xywh[:, 0] + bboxes_xywh[:, 2]).reshape(-1, 1),
                                 (bboxes_xywh[:, 1] + bboxes_xywh[:, 3]).reshape(-1, 1)))
        return xyxy_boxes
