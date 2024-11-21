import numpy as np
from sqlalchemy.testing.plugin.plugin_base import warnings

from core.utils.geometry.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:
    """
    Description:
        Class for storing and operating on bounding boxes array.
        Array of bounding boxes, represented by [top, left, width, height] format in image coordinates.

    :ivar values: bounding boxes values
    """

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['values']
    def __init__(self, bounding_boxes: np.ndarray | None = None, mode=XYWH):
        if bounding_boxes is not None:
            if mode == BoundingBoxes2DArray.XYXY:
                bounding_boxes = self.xyxy_to_xywh(bounding_boxes)
            elif not self.is_consistent(bounding_boxes):
                raise ValueError('Bounding boxes dimensions should be positive.')
            self.values = bounding_boxes
        else:
            self.values = np.empty((0, 4))


    def __getitem__(self, item):
        return self.values[item]
    
    
    def __len__(self) -> int:
        return self.values.shape[0]


    def __iter__(self):
        for index in range(self.values.shape[0]):
            yield self.values[index]


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
        if (mode == BoundingBoxes2DArray.XYWH) and (np.any(bounding_boxes[:, 2] < 0) or np.any(bounding_boxes[3] < 0)):
            raise ValueError('Input bounding_boxes[:, 2:4] components should be greater or equal zero.')
        if mode == BoundingBoxes2DArray.XYXY:
            bounding_boxes = BoundingBoxes2DArray.xyxy_to_xywh(bounding_boxes)

        self.values = np.vstack((self.values, bounding_boxes))


    def circumscribe(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Circumscribe all bounding boxes. Result is also a bounding box.

        :return: bounding box
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        top_lefts_minimum = np.min(selected_boxes[:, :2], axis=0)
        right_bottoms = selected_boxes[:, :2] + selected_boxes[:, 2:]
        right_bottoms_maximum = np.max(right_bottoms, axis=0)

        circumscribed_box = np.array([*top_lefts_minimum, *right_bottoms_maximum])
        return self.xyxy_to_xywh(circumscribed_box)


    def areas(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Calculates areas of all bounding boxes.

        :return: array of areas
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        return selected_boxes[:, 2] * selected_boxes[:, 3]


    def mean_area(self, indices: np.ndarray | None = None) -> float:
        """
        Description:
            Mean area of bounding boxes with given ``indices``.

        :param indices: indices of bounding boxes;

        :return: mean area of selected bounding boxes
        """
        areas = self.areas(indices)
        return np.mean(areas)


    def perimeters(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:
            Calculates perimeters of all bounding boxes.

        :param indices: indices of bounding boxes;

        :return: perimeters of bounding boxes
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
    def clamp_size(boxes, clamp_box):
        """
        Description:
            all ``boxes`` are inside ``clamp_box``.

        :param boxes: bounding boxes to clamp
        :param clamp_box: bounding box

        :return: clamped bounding boxes.

        :raises ValueError: If input arguments shape or size is incorrect.
        """
        clamp_box = clamp_box.flatten()
        if not (len(boxes.shape) == 2 and boxes.shape[1] == 4) or not (clamp_box.shape[0] == 4):
            raise ValueError('Boxes argument should have (N, 4) shape and clamp_box should have size of 4.')

        boxes_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(boxes)
        clamp_box_xyxy = BoundingBoxes2DArray.xywh_to_xyxy(clamp_box)

        top_lefts = np.maximum(boxes_xyxy[:, :2], clamp_box_xyxy[:, :2], axis=1)
        bottom_rights = np.minimum(boxes_xyxy[:, 2:], clamp_box_xyxy[:, 2:], axis=1)

        clamped_boxes_xyxy = np.hstack((top_lefts, bottom_rights))
        return BoundingBoxes2DArray.xyxy_to_xywh(clamped_boxes_xyxy)


    def __selected_bounding_boxes(self, indices: np.ndarray | None = None) -> np.ndarray:
        """
        Description:

        :param indices:

        :return: selected by indices bounding boxes array.

        :raises ValueError:
        """
        if indices is None:
            return self.values

        if not np.issubdtype(indices, np.int32):
            raise ValueError('Indices should be an array of integers.')
        elif len(indices.shape) != 1:
            raise ValueError('Indices should be an 1D array.')

        if indices.size > self.values.shape[0]:
            warnings.warn('Indices size bigger than the number of bounding boxes.')
            boxes_number = self.values.shape[0]
            indices = indices[:boxes_number]

        indices = indices.sort()
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
        xyxy_boxes =  np.hstack([bboxes_xywh[:, 0].reshape(-1, 1),
                                 bboxes_xywh[:, 1].reshape(-1, 1),
                                 (bboxes_xywh[:, 0] + bboxes_xywh[:, 2]).reshape(-1, 1),
                                 (bboxes_xywh[:, 1] + bboxes_xywh[:, 3]).reshape(-1, 1)])
        return xyxy_boxes
