from __future__ import annotations
import numpy as np
import os
import warnings
from typing import Type


class FramesSegments:
    """
    Description:
        Data storage class for video segments information. Video segments are ordered, means  each next segment value is greater or equal current value.

    :ivar values: array of frames segments [[segment1_frame_start, segment1_frame_end], [segment2_frame_start, segment2_frame_end], ...]
    """

    __slots__ = ['values']

    def __init__(self, segments: np.ndarray | None = None):
        if segments is not None and len(segments.shape) == 2 and segments.shape[1] == 2:
            is_consistent = FramesSegments._check_consistency(segments)
            self.values = segments if is_consistent else np.empty((0, 2))
            if not is_consistent:
                warnings.warn('Segments are not consistent. Resetting to empty shape.')
        elif segments is None:
            self.values = np.empty((0, 2))
        else:
            self.values = np.empty((0, 2))
            warnings.warn('Segments argument should be None or numpy array with 2-dimensional shape. Resetting to empty shape.')


    def __getitem__(self, item):
        return self.values[item]


    def __iter__(self):
        for index in range(self.values.shape[0]):
            yield self.values[index]


    def __len__(self) -> int:
        return self.values.shape[0]


    def __copy__(self) -> FramesSegments:
        return  FramesSegments(self.values.copy())


    def append_segment(self, segment: np.ndarray) -> None:
        """
        Description:
            Appends new segment.

        :param segment: segment to append
        """
        if isinstance(segment, np.ndarray) and segment.size != 2 :
            raise ValueError('Segment size should be 2')

        segment_flatten = segment.flatten()
        if segment_flatten[0] < self.values[-1, -1]:
            raise ValueError('Wrong segment values. Start new segment value should be greater or equal than last segment end value')

        new_segment = np.array([segment_flatten[0], segment_flatten[1]])
        self.values = np.vstack((self.values, new_segment))


    def filter_by_length(self, threshold: int) -> None:
        """
        Description:
            Filter segments by their lengths in place. Segments with length less or equal than ``threshold`` will be deleted.

        :param threshold: segment length threshold
        """
        segments_lengths =  np.abs(self.values[:, 1] - self.values[:, 0])
        segments_mask = segments_lengths > threshold
        self.values = self.values[segments_mask]


    def filter_degenerate(self) -> None:
        """
        Description:
            In some cases we may obtain segments with sero length, e.g. [x_0, x_0]. This function filters them out.
        """
        self.filter_by_length(0)


    def complement(self, lower_bound: int, upper_bound: int, *args, **kwargs) -> None:
        r"""
        Description:
            Video segments complement set closure, where set is a  :math:`[0, N_{f} - 1]` segment. Formula:

            .. math::
                \mathbf{C} \Big \{ [0, N_f - 1]  \ \backslash  \  \left ( \cup_{n=1}^{N_s} s_n \right) \Big \}

            where :math:`N_f` -- number of frames, :math:`N_s` -- number of segments, :math:`\{s_n\}` -- segments,
            :math:`\mathbf{C}` -- set closure.

        :param lower_bound: lower bound of a set, containing all segments.
        :param upper_bound: high bound of a set, containing all segments.

        :raises ValueError: when lower or high bound has incorrect values.

        :return: video segments complement
        """
        if self.values.shape[0] == 0:
            self.values = np.array([[lower_bound, upper_bound]])
            return
        elif lower_bound > self.values[0, 0]:
            raise ValueError('Lower bound is greater than the start of the first segment')
        elif upper_bound < self.values[-1, -1]:
            raise ValueError('Upper bound is less then the end of the last segment.')

        self.values = self.values.flatten()
        self.values = np.insert(self.values, 0, lower_bound)
        self.values = np.append(self.values, upper_bound)
        self.values = self.values.reshape(-1, 2)

        if self.values[0, 0] == self.values[0, 1]:
            self.values = np.delete(self.values, 0, axis=0)
        if self.values[-1, 0] == self.values[-1, 1]:
            self.values = np.delete(self.values, -1, axis=0)


    def bridge_gaps(self, gap_threshold: int) -> None:
        """
        Description:
            Bridge gaps in place between segments if gaps itself less than ``gaps_length``.

        :param gap_threshold: maximum gap length
        """
        if self.values.shape[0] <= 1:
            return

        low_bound = int(self.values[0, 0])
        high_bound = int(self.values[-1, -1])

        self.complement(low_bound, high_bound)
        self.filter_by_length(gap_threshold)
        self.complement(low_bound, high_bound)


    def combine_adjacent(self) -> None:
        """
        Description:
            Combine in place adjacent segments. E.g. segments [0, 200] and [200, 599] will be combined to [0, 599] segment.
        """
        self.bridge_gaps(0)


    def clip(self, minimum, maximum) -> FramesSegments:
        """
        Description:
            Clip segments to given ``minimum`` and ``maximum`` vales.
            If the whole segment iii less than ``minimum`` or greater than ``maximum``value, it will not be included in the result.

        :param minimum: minimum frame value
        :param maximum: maximum frame value

        :return: clipped frames segments
        """
        minimum_mask = np.argwhere(self.values >= minimum)
        maximum_mask = np.argwhere(self.values <= maximum)
        clip_2d_mask = np.logical_and(minimum_mask, maximum_mask)
        clip_mask = np.logical_or(clip_2d_mask[:, 0], clip_2d_mask[:, 1])
        clipped_values = self.values[clip_mask]
        return FramesSegments(clipped_values)


    def write(self, filepath: str) -> None:
        """
        Description:
            Write segments to file using numpy.write() function.

        :raises OSError: if ``filepath`` does not exist.
        """
        directory_name = os.path.dirname(filepath)
        if os.path.exists(directory_name):
            np.save(filepath, self.values)
        else:
            raise OSError('Filepath directory does not exist.')


    @staticmethod
    def _check_consistency(segments: np.ndarray) -> bool:
        """
        Description:
            Check segments consistency. This means all segments endpoints are in non-decreasing order.
            If [x_1, x_2], [x_3, x_4] .... [x_N, x_N+1] - segments, then x_1 <= x_2 <= x_3 <= x_4 <= ... <= x_N <= X_N+1.

        :return: True if segments are consistent, False otherwise.
        """
        segments_flat_points = segments.flatten()
        segments_points_difference = segments_flat_points[1:] - segments_flat_points[:-1]
        if np.all(segments_points_difference >= 0):
            return True
        return False


    @property
    def size(self) -> int:
        """
        Description:
            Returns segments size.

        :return: segments size
        """
        return self.values.size


    @property
    def shape(self) -> tuple:
        """
        Description:
            Returns segments shape.

        :return: segments shape
        """
        return self.values.shape


    @property
    def lengths(self) -> np.ndarray:
        """
        Description:
            Returns segments lengths.

        :return: segments lengths.
        """
        return np.abs(self.values[:, 1] - self.values[:, 0])


    def frames_indices(self) -> list[Type[np.ndarray]]:
        """
        Description:
            Calculates frames indices. If [f_start, f_end] is a frame segment, frames indices will be [f_start, f_start + 1, ..., f_end - 1]

        :return: frames indices
        """
        number_segments = self.values.shape[0]
        frames_indices = [np.ndarray] * number_segments

        for index in range(number_segments):
            current_segment = self.values[index]
            elements_number = current_segment[1] - current_segment[0]
            frames_indices[index] = np.linspace(current_segment[0], current_segment[1], elements_number, endpoint=False).astype(np.int64)

        return frames_indices
