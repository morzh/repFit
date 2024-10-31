import numpy as np
from sqlalchemy.testing.plugin.plugin_base import warnings

from core.utils.geometry.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['bounding_boxes']
    def __init__(self, bounding_boxes: np.ndarray | None = None):
        self.bounding_boxes: np.ndarray = np.empty((0, 4))


    def append(self, bounding_box: np.ndarray, mode=XYWH):
        """
        Description:
            Append new bounding box.
        """
        if bounding_box.size == 4:
            bounding_box = bounding_box.flatten()
            self.bounding_boxes = np.vstack((self.bounding_boxes, bounding_box))
        else:
            warnings.warn('Input bounding_box size should be 4')

    def circumscribe(self) -> BoundingBox2D:
        """
        Description:
            Circumscribe all bounding boxes. Result is also a bounding box.

        :return: bounding box
        """
        top_lefts_minimum = np.min(self.bounding_boxes[:, :2], axis=0)
        right_bottoms = self.bounding_boxes[:, :2] + self.bounding_boxes[:, 2:]
        right_bottoms_maximum = np.max(right_bottoms, axis=0)

        return BoundingBox2D(top_lefts_minimum[0], top_lefts_minimum[1], right_bottoms_maximum[0], right_bottoms_maximum[1], mode=self.XYXY)

    def areas(self) -> np.ndarray:
        """
        Description:
            Calculates areas of all bounding boxes.

        :return: array of areas
        """
        return self.bounding_boxes[:, 2] * self.bounding_boxes[:, 3]


    def perimeters(self) -> np.ndarray:
        """
        Description:
            Calculates perimeters of all bounding boxes.

        :return: perimeters of bounding boxes
        """
        return 2 * self.bounding_boxes[:, 2] + self.bounding_boxes[:, 3]