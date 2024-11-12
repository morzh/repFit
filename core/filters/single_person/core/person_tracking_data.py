import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.strictly_increasing_sequence import StrictlyIncreasingSequence
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class PersonTrackingData:
    """
    Description:
        Class for storing  data, obtained from person's tracker (neural network actually).

    :ivar _bounding_boxes: person's bounding boxes;
    :ivar _frames_indices: frame indices at which person was tracked;
    :ivar _confidences: tracking confidences.
    """
    __slots__ = ['_bounding_boxes', '_frames_indices', '_confidences']

    def __init__(self):
        self._bounding_boxes = BoundingBoxes2DArray()
        self._frames_indices = StrictlyIncreasingSequence()
        self._confidences: np.ndarray = np.empty(0, )


    def append(self, bounding_box: np.ndarray, frame_index: int, confidence: float, bounding_box_mode=BoundingBoxes2DArray.XYWH) -> None:
        """
        Description:
            Append new tracked data.

        :param bounding_box: bounding box to append
        :param frame_index: frame index to append
        :param confidence: confidence to append
        :param bounding_box_mode: bounding box mode (XYWH or XYXY)

        :raise ValueError: if one of ``bounding_box`` or ``frame_index`` or ``confidence`` values is incorrect.
        """
        self._bounding_boxes.append(bounding_box, mode=bounding_box_mode)
        self._frames_indices.append(frame_index)

        if confidence >= 0:
            self._confidences = np.append(self._confidences, confidence)
        else:
            raise ValueError('Confidence should be greater or equal zero')


    def bounding_box(self, frame_index) -> np.ndarray:
        """
        Description:
            Calculates bounding box at ``frame_index``. If ``frame_index`` presented in data, corresponding bounding box will be returned.
            In other case, bounding box will be interpolated.

        :param frame_index: frame index

        :return: bounding box, represented by numpy array
        """
        index_high_bound_occurrence = np.argmax(self._frames_indices >= frame_index)

        if self._frames_indices[index_high_bound_occurrence] == frame_index:
            return self._bounding_boxes[index_high_bound_occurrence]
        else:
            segment_start = self._frames_indices[index_high_bound_occurrence - 1]
            segment_end = self._frames_indices[index_high_bound_occurrence]
            factor = (frame_index - segment_start) / (segment_end - segment_start)
            return self._bounding_boxes[index_high_bound_occurrence - 1] + factor * (self._bounding_boxes[index_high_bound_occurrence] - self._bounding_boxes[index_high_bound_occurrence - 1])


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
        Description:
            Checks if data is consistent

        :return: True if consistent, False otherwise.
        """
        return StrictlyIncreasingSequence.is_consistent(self._frames_indices) and BoundingBoxes2DArray.is_consistent(self._bounding_boxes.values)


    @property
    def bounding_boxes(self) -> BoundingBoxes2DArray:
        """
        Description:
            Get bounding boxes data.
        """
        return self._bounding_boxes


    @property
    def frames_indices(self) -> StrictlyIncreasingSequence:
        """
        Description:
            Get frames indices data.
        """
        return self._frames_indices


    @property
    def confidences(self) -> np.ndarray:
        """
        Description:
            Get  confidences data.
        """
        return self._confidences
