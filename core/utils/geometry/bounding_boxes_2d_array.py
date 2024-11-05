import numpy as np

from core.utils.geometry.bounding_box_mode import BoundingBoxMode
from core.utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['bounding_boxes']
    def __init__(self, bounding_boxes: np.ndarray | None = None):
        self.bounding_boxes: np.ndarray = np.empty((0, 4))


    def append(self, bounding_box: np.ndarray, interpolation_steps = 1, mode=XYWH):
        """
        Description:
            Append new bounding box.

        :param bounding_box:
        :param interpolation_steps:
        :param mode:
        """
        if bounding_box.size != 4:
            raise ValueError('Input bounding_box size should be 4')
        if interpolation_steps < 1:
            raise ValueError('interpolation_steps should be greater or equal to one.')

        if mode == BoundingBoxes2DArray.XYXY:
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2] - bounding_box[0]
            h = bounding_box[3] - bounding_box[1]
            bounding_box = np.array([x, y, w, h])

        if interpolation_steps > 1:
            boxes_difference = bounding_box - self.bounding_boxes[-1]
            interpolation_step_value = boxes_difference / interpolation_steps
            new_bounding_boxes = np.empty((0, 4))
            for index in range(interpolation_steps):
                current_bounding_box = self.bounding_boxes[-1] + index * interpolation_step_value
                new_bounding_boxes = np.vstack((new_bounding_boxes, current_bounding_box))
        else:
            new_bounding_boxes = bounding_box.reshape((1, 4))

        self.bounding_boxes = np.vstack((self.bounding_boxes, new_bounding_boxes))

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


    def mean_area(self) -> float:
        areas = self.areas()
        return np.mean(areas)


    def perimeters(self) -> np.ndarray:
        """
        Description:
            Calculates perimeters of all bounding boxes.

        :return: perimeters of bounding boxes
        """
        return 2 * self.bounding_boxes[:, 2] + self.bounding_boxes[:, 3]