from dataclasses import dataclass
import numpy as np

from bounding_box_mode import BoundingBoxMode
from utils.geometry.bounding_box_2d import BoundingBox2D


class BoundingBoxes2DArray:

    XYWH = BoundingBoxMode.XYWH.value
    XYXY = BoundingBoxMode.XYXY.value

    __slots__ = ['boxes']
    def __init__(self):
        self.boxes: np.ndarray = np.empty((0, 4))

    def append(self, bounding_box, mode=XYWH):
        """

        """

    def circumscribe(self) -> BoundingBox2D:
        """
        Description:
        """

    def areas(self) -> np.ndarray:
        """
        Description:
        """

    def perimeters(self) -> np.ndarray:
        """
        Description:
        """