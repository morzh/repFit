from dataclasses import dataclass
import numpy as np

from utils.geometry.bounding_box_2d import BoundingBox2D


@dataclass
class BoundingBoxes2DArray:
    boxes: np.ndarray


    def append(self, bounding_vox, mode=XYWH):
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