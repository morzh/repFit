import numpy as np

from core.utils.geometry.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:
    """
    Description:

    """

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['bounding_boxes']
    def __init__(self, bounding_boxes: np.ndarray | None = None):
        self.bounding_boxes: np.ndarray = np.empty((0, 4))

    def __getitem__(self, item):
        return self.bounding_boxes[item]


    def append(self, bounding_box: np.ndarray, mode=XYWH) -> None:
        """
        Description:
            Append new bounding box to an existing array.

        :param bounding_box:
        :param mode:
        """
        if bounding_box.size != 4:
            raise ValueError('Input bounding_box size should be 4')
        if (mode == BoundingBoxes2DArray.XYWH) and (bounding_box[2] < 0 or bounding_box[3] < 0):
            raise ValueError('Input bounding_box[2:4] components should be greater or equal zero.')
        if mode == BoundingBoxes2DArray.XYXY:
            bounding_box = BoundingBoxes2DArray.xyxy_to_xywh(bounding_box)

        bounding_box = bounding_box.reshape((1, 4))
        self.bounding_boxes = np.vstack((self.bounding_boxes, bounding_box))


    def extend(self, bounding_boxes: np.ndarray, mode=XYWH):
        """
        Description:
            Add new bounding boxes to an existing array.

        :param bounding_boxes:
        :param mode:
        """
        if bounding_boxes.shape[1] != 4 or len(bounding_boxes.shape) != 2:
            raise ValueError('Input bounding_box(es) should have size [N, 4].')
        if (mode == BoundingBoxes2DArray.XYWH) and (np.any(bounding_boxes[:, 2] < 0) or np.any(bounding_boxes[3] < 0)):
            raise ValueError('Input bounding_boxes[:, 2:4] components should be greater or equal zero.')
        if mode == BoundingBoxes2DArray.XYXY:
            bounding_boxes = BoundingBoxes2DArray.xyxy_to_xywh(bounding_boxes)

        self.bounding_boxes = np.vstack((self.bounding_boxes, bounding_boxes))


    def circumscribe(self, indices: np.ndarray | None = None) -> BoundingBox2D:
        """
        Description:
            Circumscribe all bounding boxes. Result is also a bounding box.

        :return: bounding box
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        top_lefts_minimum = np.min(selected_boxes[:, :2], axis=0)
        right_bottoms = selected_boxes[:, :2] + selected_boxes[:, 2:]
        right_bottoms_maximum = np.max(right_bottoms, axis=0)

        return BoundingBox2D(top_lefts_minimum[0], top_lefts_minimum[1], right_bottoms_maximum[0], right_bottoms_maximum[1], mode=self.XYXY)


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


    def perimeters(self, indices) -> np.ndarray:
        """
        Description:
            Calculates perimeters of all bounding boxes.

        :param indices: indices of bounding boxes;

        :return: perimeters of bounding boxes
        """
        selected_boxes = self.__selected_bounding_boxes(indices)
        return selected_boxes[:, 2] + selected_boxes[:, 3]


    @property
    def shape(self) -> tuple:
        return self.bounding_boxes.shape

    @property
    def size(self) -> int:
        return self.bounding_boxes.size


    def is_consistent(self):
        """
        Description:
            Checks if 2nd and 3rd components are greater than zero.
        """
        columns_number = self.bounding_boxes.shape[1]
        return np.alltrue(self.bounding_boxes[:, 2:] > 0) and columns_number == 4


    def __selected_bounding_boxes(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            selected_boxes = self.bounding_boxes
        elif len(indices.shape) != 1:
            raise ValueError('Indices should be an 1D array.')
        else:
            selected_boxes = self.bounding_boxes[indices]

        return selected_boxes


    @staticmethod
    def xyxy_to_xywh(bounding_box_xyxy: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.
        """

        xyxy = bounding_box_xyxy.reshape((-1, 2, 2))
        xy_top_left = np.min(xyxy, axis=2)
        xy_bottom_right = np.max(xyxy, axis=0)
        xywh_bounding_box = np.hstack((xy_top_left, xy_bottom_right - xy_top_left))

        return xywh_bounding_box.reshape((-1, 4))


    @staticmethod
    def xywh_to_xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.
        """
        xyxy =  np.array([bbox_xywh[:, 0], bbox_xywh[:, 1], bbox_xywh[:, 0] + bbox_xywh[:, 2], bbox_xywh[:, 1] + bbox_xywh[:, 3]])
        return xyxy
