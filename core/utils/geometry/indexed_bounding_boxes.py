import numpy as np

from core.utils.cv.segments import Segments
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class IndexedBoundingBoxes:
    """
    Description:

    """
    __slots__ = ['bounding_boxes', 'indices']
    def __init__(self):
        self.bounding_boxes = BoundingBoxes2DArray()
        self.indices: np.ndarray = np.empty(0,)


    def append(self, bounding_box: np.ndarray, index:int, mode=BoundingBoxes2DArray.XYWH) -> None:
        self.bounding_boxes.append(bounding_box,  mode=mode)
        self.indices = np.append(self.indices, index)


    def bounding_box(self, index) -> np.ndarray:
        index_occurrence = np.argmax(self.indices == index)
        if index_occurrence is None:
            index_occurrence = np.argmax(self.indices > index)

        if index_occurrence == 0:
            return self.bounding_boxes[0]
        else:
            return 0.5 * (self.bounding_boxes[index_occurrence - 1] + self.bounding_boxes[index_occurrence])


    def calculate_segments(self, stride=1) -> Segments:
        segments_bins = np.hstack((self.indices.reshape(-1, 1), self.indices.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = np.nan

        mask  = segments_bins[:, 0] != np.nan
        segments = segments_bins[mask]
        segments[:, 1] += 1
        return Segments(segments)


    def _is_consistent(self) -> bool:
        derivative = self.indices[1:] - self.indices[-1:]
        are_indices_strictly_increasing = np.alltrue(derivative > 0)
        are_shapes_equal = self.bounding_boxes.shape[0] == self.indices.shape[0]
        return are_indices_strictly_increasing and are_shapes_equal and self.bounding_boxes.is_consistent()
