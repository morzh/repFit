import numpy as np

from core.utils.geometry.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:
    """
    Description:
        Class for storing and operating on bounding boxes array.
        Array of bounding boxes, represented by [top, left, width, height] format in image coordinates.
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


    def append(self, bounding_box: np.ndarray, mode=XYWH) -> None:
        """
        Description:
            Append new bounding box to an existing array.

        :param bounding_box:
        :param mode:
        """
        if bounding_box.size != 4:
            raise ValueError('Input bounding_box size should be 4')
        bounding_box_flatten = bounding_box.flatten()
        if (mode == BoundingBoxes2DArray.XYWH) and (bounding_box_flatten[2] < 0 or bounding_box_flatten[3] < 0):
            raise ValueError('Input bounding_box[2:4] components should be greater or equal zero.')
        if mode == BoundingBoxes2DArray.XYXY:
            bounding_box = BoundingBoxes2DArray.xyxy_to_xywh(bounding_box)

        bounding_box = bounding_box.reshape((1, 4))
        self.values = np.vstack((self.values, bounding_box))


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
        # return BoundingBox2D(top_lefts_minimum[0], top_lefts_minimum[1], right_bottoms_maximum[0], right_bottoms_maximum[1], mode=self.XYXY)


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
        return self.values.shape

    @property
    def size(self) -> int:
        return self.values.size

    @staticmethod
    def is_consistent(bounding_boxes: np.ndarray):
        """
        Description:
            Checks if 2nd and 3rd components are greater than zero.
        """
        columns_number = bounding_boxes.shape[1]
        return np.alltrue(bounding_boxes[:, 2:] > 0) and columns_number == 4


    def __selected_bounding_boxes(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            selected_boxes = self.values
        elif len(indices.shape) != 1:
            raise ValueError('Indices should be an 1D array.')
        else:
            selected_boxes = self.values[indices]

        return selected_boxes


    @staticmethod
    def xyxy_to_xywh(bounding_box_xyxy: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.
        """
        # TODO: process degenerate cases
        xyxy = bounding_box_xyxy.reshape((-1, 2, 2))
        xy_top_left = np.min(xyxy, axis=1)
        xy_bottom_right = np.max(xyxy, axis=1)
        xywh_boxes = np.hstack((xy_top_left, xy_bottom_right - xy_top_left))

        return xywh_boxes.reshape((-1, 4))


    @staticmethod
    def xywh_to_xyxy(bbox_xywh: np.ndarray) -> np.ndarray:
        """
        Description:
            Converts XYXY (top-left, bottom-right) bounding box representation to XYWH (top-left, width height) representation.
        """
        xyxy_boxes =  np.hstack([bbox_xywh[:, 0].reshape(-1, 1),
                                 bbox_xywh[:, 1].reshape(-1, 1),
                                (bbox_xywh[:, 0] + bbox_xywh[:, 2]).reshape(-1, 1),
                                (bbox_xywh[:, 1] + bbox_xywh[:, 3]).reshape(-1, 1)])
        return xyxy_boxes
