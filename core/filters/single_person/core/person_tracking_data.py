import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.increasing_sequence import StrictlyIncreasingSequence
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class PersonTrackingData:
    """
    Description:

    """
    __slots__ = ['_bounding_boxes', '_frames_indices', '_confidences']

    def __init__(self):
        self._bounding_boxes = BoundingBoxes2DArray()
        self._frames_indices = StrictlyIncreasingSequence()
        self._confidences: np.ndarray = np.empty(0, )


    def append(self, bounding_box: np.ndarray, frame_index:int, confidence: float, bounding_box_mode=BoundingBoxes2DArray.XYWH) -> None:
        """
        Description:

        :param bounding_box:
        :param frame_index:
        :param confidence:
        :param bounding_box_mode:

        :raise ValueError:
        """
        self._bounding_boxes.append(bounding_box, mode=bounding_box_mode)
        self._frames_indices.append(frame_index)
        self._confidences = np.append(self._confidences, confidence)


    def bounding_box(self, index) -> np.ndarray:
        """
        Description:

        :param index:

        :return: bounding box, represented by numpy array
        """
        index_occurrence = np.argmax(self._frames_indices == index)
        if index_occurrence is None:
            index_occurrence = np.argmax(self._frames_indices > index)

        if index_occurrence == 0:
            return self._bounding_boxes[0]
        else:
            return 0.5 * (self._bounding_boxes[index_occurrence - 1] + self._bounding_boxes[index_occurrence])


    def calculate_segments(self, stride=1) -> FramesSegments:
        """
        Description:

        :param stride:

        :return: frame segments
        """
        segments_bins = np.hstack((self._frames_indices.values.reshape(-1, 1), self._frames_indices.values.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = np.nan

        mask = segments_bins[:, 0] != np.nan
        segments = segments_bins[mask]
        segments[:, 1] += 1
        return FramesSegments(segments)


    def _is_consistent(self) -> bool:
        """

        """
        return self._frames_indices.is_consistent() and BoundingBoxes2DArray.is_consistent(self._bounding_boxes.values)


    @property
    def bounding_boxes(self) -> BoundingBoxes2DArray:
        """
        Description:

        """
        return self._bounding_boxes


    @property
    def frames_indices(self) -> StrictlyIncreasingSequence:
        """
        Description:

        """
        return self._frames_indices


    @property
    def confidences(self) -> np.ndarray:
        """
        Description:

        """
        return self._confidences
