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


    def __len__(self):
        return self.frames_indices.values.shape[0]


    def __delitem__(self, index):
        self._confidences =np.delete(self._confidences, index)
        self._bounding_boxes.values = np.delete(self._bounding_boxes.values, index, axis=0)
        self._frames_indices._values = np.delete(self._frames_indices.values, index)


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
            In other case, bounding box will be interpolated or extrapolated.

        :param frame_index: frame index

        :raises ValueError:

        :return: bounding box, represented by numpy array
        """
        if len(self._frames_indices) == 0:
            return np.array([-1, -1, 0, 0])
        elif len(self._frames_indices) == 1:
            return self._bounding_boxes[0]
        elif frame_index <= self._frames_indices.values[-1]:
            return self._bounding_box_interpolation(frame_index)
        else:
            return self._bounding_box_extrapolation(frame_index)


    def confidence(self, frame_index) -> float:
        """
        Description:
            Returns confidence value at given ``frame_index``.
            Under the hood, it finds nearest to input ``frame_index`` confidence value.

        :param frame_index: frame index

        :return: confidence value
        """
        if len(self._frames_indices) == 0:
            return 0.0
        elif len(self._frames_indices) == 1:
            return float(self._confidences[0])

        confidence_index = np.argmin(np.abs(self._frames_indices.values - frame_index))
        confidence_value = float(self._confidences[confidence_index])
        return confidence_value


    def calculate_segments(self, stride=1) -> FramesSegments:
        """
        Description:
            Calculate frames segments from stored data.

        :param stride: frames stride

        :return: frame segments
        """
        segments_bins = np.hstack((self._frames_indices.values.reshape(-1, 1), self._frames_indices.values.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = -1

        mask = segments_bins[:, 0] != -1
        segments = segments_bins[mask]
        segments[:, 1] += 1
        return FramesSegments(segments)


    def _is_consistent(self) -> bool:
        """
        Description:
            Checks if class instance data is consistent.

        :return: True if consistent, False otherwise.
        """
        return StrictlyIncreasingSequence.is_consistent(self._frames_indices.values) and BoundingBoxes2DArray.is_consistent(self._bounding_boxes.values)


    def _bounding_box_interpolation(self, frame_index: int) -> np.ndarray:
        """
        Description:
        """
        index_high_bound_occurrence = np.argmax(self._frames_indices.values >= frame_index)

        if self._frames_indices[index_high_bound_occurrence] == frame_index:
            return self._bounding_boxes[index_high_bound_occurrence]
        else:
            segment_start = self._frames_indices[index_high_bound_occurrence - 1]
            segment_end = self._frames_indices[index_high_bound_occurrence]
            factor = (frame_index - segment_start) / (segment_end - segment_start)

            return self._bounding_boxes[index_high_bound_occurrence - 1] + factor * (self._bounding_boxes[index_high_bound_occurrence] - self._bounding_boxes[index_high_bound_occurrence - 1])


    def _bounding_box_extrapolation(self, frame_index) -> np.ndarray:
        """
        Description:
        """
        last_frame_index = self._frames_indices.values[-1]
        direction = self._bounding_boxes[-1] - self._bounding_boxes[-2]
        factor = frame_index - last_frame_index
        extrapolated_bounding_box = self._bounding_boxes[-1] + factor * direction
        # print(extrapolated_bounding_box)
        return extrapolated_bounding_box


    @property
    def bounding_boxes(self) -> BoundingBoxes2DArray:
        """
        Description:
            Bounding boxes getter.

        :return: bounding boxes
        """
        return self._bounding_boxes


    @property
    def frames_indices(self) -> StrictlyIncreasingSequence:
        """
        Description:
            Frames indices getter.

        :return: frames indices sequence.
        """
        return self._frames_indices


    @property
    def confidences(self) -> np.ndarray:
        """
        Description:
            Confidences getter.

        :return: confidences values.
        """
        return self._confidences
