import numpy as np

from core.utils.cv.segments import Segments
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class IndexedBoundingBoxes(BoundingBoxes2DArray):
    """
    Description:

    """

    __slots__ = ['indices']
    def __init__(self):
        super().__init__()
        self.indices: np.ndarray = np.empty(0,)

    def __getitem__(self, item):
        ...

    def calculate_segments(self, stride=1) -> Segments:
        ...


