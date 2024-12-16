import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.strictly_increasing_sequence import StrictlyIncreasingSequence
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class PersonTrackingData:
    """
    Description:
        Class for storing  data, obtained from person's tracker (neural network actually).

    :ivar _bounding_boxes: person's bounding boxes;
    :ivar _frames_indices: frame indices at which person was tracked;
    :ivar _confidences: person tracking confidences.
    :ivar _keypoints: person 2D keypoints with confidences
    """
    __slots__ = ['_bounding_boxes', '_frames_indices', '_confidences', '_keypoints']

    def __init__(self):
        self._bounding_boxes = BoundingBoxes2DArray()
        self._frames_indices = StrictlyIncreasingSequence()
        self._confidences = np.empty(0, )
        self._keypoints = np.empty((0, 17, 3))


    def __len__(self):
        return self.frames_indices.values.shape[0]


    def __delitem__(self, index):
        self._confidences =np.delete(self._confidences, index)
        self._bounding_boxes.values = np.delete(self._bounding_boxes.values, index, axis=0)
        self._frames_indices._values = np.delete(self._frames_indices.values, index)
        self._keypoints = np.delete(self._keypoints, index, axis=0)


    def append(self, bounding_box: np.ndarray, frame_index: int, confidence: float, bounding_box_mode=BoundingBoxes2DArray.XYWH, keypoints: np.ndarray | None = None) -> None:
        """
        Description:
            Append new tracked data.

        :param bounding_box: bounding box to append
        :param frame_index: frame index to append
        :param confidence: confidence to append
        :param bounding_box_mode: bounding box mode (XYWH or XYXY)
        :param keypoints: key points to append

        :raise ValueError: if one of ``bounding_box`` or ``frame_index`` or ``confidence`` values is incorrect.
        """
        self._bounding_boxes.append(bounding_box, mode=bounding_box_mode)
        self._frames_indices.append(frame_index)

        if keypoints is not None:
            self._keypoints = np.vstack((self._keypoints, np.expand_dims(keypoints, axis=0)))

        if confidence >= 0:
            self._confidences = np.append(self._confidences, confidence)
        else:
            raise ValueError('Confidence should be greater or equal zero')


    def apply_mask(self, mask: np.ndarray) -> None:
        """
        Description:
            Apply 1D mask to tracking data.

        :param mask: mask
        """
        if mask.shape[0] != len(self._frames_indices):
            raise ValueError('Mask shape should be (N,), where N is frame indices number')
        elif not mask.dtype == bool:
            raise ValueError('Mask values should be boolean.')

        self._keypoints = self._keypoints[mask]
        self._bounding_boxes.values = self._bounding_boxes[mask]
        self._confidences = self._confidences[mask]
        self._frames_indices._values = self._frames_indices[mask]


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
            return self._interpolation(frame_index, self._bounding_boxes)
        else:
            return self._extrapolation(frame_index, self._bounding_boxes)


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

    def frame_keypoints(self, frame_index: int) -> np.ndarray:
        """
        Description:
            Calculates keypoints at ``frame_index``.

        :param frame_index: index of a frame;

        :return: keypoints at given ``frame''
        """
        if len(self._frames_indices) == 0:
            return np.empty((17, 3))
        elif len(self._frames_indices) == 1:
            return self._keypoints[0]
        elif frame_index <= self._frames_indices.values[-1]:
            return self._coco_keypoints_interpolation(frame_index, self._keypoints)
        else:
            return self._extrapolation(frame_index, self._keypoints)


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


    def _coco_keypoints_interpolation(self, frame_index: int, keypoints: np.ndarray) -> np.ndarray:
        """
        Description:
            Interpolate ``keypoints`` at ``frame_index``.
            YOLO assigns zero coordinates if joint cannot be tracked. To account this,
            zero value is set to interpolated keypoint from any of the input values. ..... BLAH BLAH ...

        :param frame_index: index of a frame
        :param keypoints: COCO keypoints array

        :return: interpolated keypoints
        """
        index_occurrence = np.argmax(self._frames_indices.values >= frame_index)

        if self._frames_indices[index_occurrence] == frame_index:
            return keypoints[index_occurrence]
        else:
            frame_1 = self._frames_indices[index_occurrence - 1]
            frame_2 = self._frames_indices[index_occurrence]
            factor = (frame_index - frame_1) / (frame_2 - frame_1)

            value_1 = keypoints[index_occurrence - 1]
            value_2 = keypoints[index_occurrence]

            zeros_mask = np.vstack((np.argwhere(value_1 < 1e-6), np.argwhere(value_2 < 1e-6)))
            zeros_mask = np.unique(zeros_mask, axis=0)

            interpolated_keypoints = value_1 + factor * (value_2 - value_1)
            interpolated_keypoints[zeros_mask[:, 0], zeros_mask[:, 1]] = 0.0

            return interpolated_keypoints


    def _coco_keypoints_extrapolation(self, frame_index, keypoints) -> np.ndarray:
        """
        Description:
            Extrapolate ``array`` at ``frame_index``.
            YOLO assigns zero coordinates if joint cannot be tracked. To account this,
            zero value is set to extrapolated keypoint from any of the input values. ..... BLAH BLAH

        :param frame_index: index of a frame
        :param keypoints: COCO keypoints array

        :return: extrapolated keypoints.
        """
        last_frame_index = self._frames_indices.values[-1]
        direction = keypoints[-1] - keypoints[-2]
        factor = frame_index - last_frame_index

        zeros_mask = np.vstack((np.argwhere(keypoints[-1] < 1e-6), np.argwhere(keypoints[-2] < 1e-6)))
        zeros_mask = np.unique(zeros_mask, axis=0)

        extrapolated_keypoints = keypoints[-1] + factor * direction
        extrapolated_keypoints[zeros_mask[:, 0], zeros_mask[:, 1]] = 0.0
        return extrapolated_keypoints


    def _interpolation(self, frame_index: int, array) -> np.ndarray:
        """
        Description:
            Interpolate ``array`` at ``frame_index``.

        :param frame_index: index of a frame
        :param array:

        :return: interpolated values
        """
        index_occurrence = np.argmax(self._frames_indices.values >= frame_index)

        if self._frames_indices[index_occurrence] == frame_index:
            return array[index_occurrence]
        else:
            segment_start = self._frames_indices[index_occurrence - 1]
            segment_end = self._frames_indices[index_occurrence]
            factor = (frame_index - segment_start) / (segment_end - segment_start)

            return array[index_occurrence - 1] + factor * (array[index_occurrence] - array[index_occurrence - 1])


    def _extrapolation(self, frame_index, array) -> np.ndarray:
        """
        Description:
            Extrapolate ``array`` at ``frame_index``.

        :param frame_index:
        :param array:

        :return: extrapolated values
        """
        last_frame_index = self._frames_indices.values[-1]
        direction = array[-1] - array[-2]
        factor = frame_index - last_frame_index
        extrapolated_array = array[-1] + factor * direction
        return extrapolated_array


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

    @property
    def keypoints(self) -> np.ndarray:
        """
        Description:
            Person keypoints getter

        :return: keypoints
        """
        return self._keypoints

    @property
    def keypoints_confidences(self) -> np.ndarray:
        """
        Description:
            Person keypoints confidences getter

        :return: keypoints confidences
        """
        return self._keypoints[:, 2]

    @property
    def joints_number(self):
        return self._keypoints.shape[1]
