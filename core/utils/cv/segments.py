import copy
import numpy as np
import os
import warnings

from typing import Self


class Segments:
    """
    Description:
        Data storage class for video segments information. Video segments are ordered, means  each next segment value is greater or equal current value.

    :ivar segments: array of frames segments [[segment1_frame_start, segment1_frame_end], [segment2_frame_start, segment2_frame_end], ...]
    """
    __slots__ = ['segments']
    def __init__(self, segments: np.ndarray | None = None):
        if len(segments.shape) == 2 and segments.shape[1] == 2:
            is_consistent = Segments._check_consistency(segments)
            self.segments = segments if is_consistent else np.empty((0, 2))
            warnings.warn('Segments are not consistent. Resetting to empty shape.')
        elif segments is None:
            self.segments = np.empty((0, 2))
        else:
            self.segments = np.empty((0, 2))
            warnings.warn('Segments argument should be None or numpy array with 2-dimensional shape. Resetting to empty shape.')


    def __getitem__(self, item):
        return self.segments[item]


    def append_segment(self, segment: np.ndarray | tuple[int, int]) -> None:
        """
        Description:
            Appends new segment.

        :param segment: segment to append
        """
        if segment.size != 2:
            raise ValueError('Segment size should be 2')
        segment = np.array([segment[0], segment[1]])  # TODO: account different numpy shapes
        self.segments = np.vstack((self.segments, segment))


    def filter_by_length(self, threshold: int) -> None:
        """
        Description:
            Filter segments by their lengths in place. Segments with length less or equal than ``threshold`` will be deleted.

        :param threshold: segment length threshold
        """
        for segment_index, current_segment in enumerate(self.segments):
            current_segment_length = current_segment[1] - current_segment[0]
            if current_segment_length < threshold:
                self.segments[segment_index] = np.array([-1, -1])
        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]


    def complement(self, frames_number: int, *args, **kwargs) -> Self:
        r"""
        Description:
            Video segments complement set closure, where set is a  :math:`[0, N_{f} - 1]` segment. Formula:

            .. math::
                \mathbf{C} \Big \{ [0, N_f - 1]  \ \backslash  \  \left ( \cup_{n=1}^{N_s} s_n \right) \Big \}

            where :math:`N_f` -- number of frames, :math:`N_s` -- number of segments, :math:`\{s_n\}` -- segments,
            :math:`\mathbf{C}` -- set closure.
        :return:  video segments complement
        """
        segments = self.segments.flatten()
        segments = np.insert(segments, 0, 0)
        segments = np.append(segments, frames_number - 1)
        segments = segments.reshape(-1, 2)

        if segments[0, 0] == segments[0, 1]:
            segments = np.delete(segments, 0, axis=0)
        if segments[-1, 0] == segments[-1, 1]:
            segments = np.delete(segments, -1, axis=0)

        video_segments_complement = copy.copy(self)
        video_segments_complement.segments = segments

        return video_segments_complement

    def bridge_gaps(self, gap_length: int):
        """
        Description:
            Bridge gaps in place between segments if gaps itself less than ``gaps_length``.

        :param gap_length: gap length
        """
        segments_flatten = self.segments.flatten()
        segments_gaps_flatten = segments_flatten[1:-1]
        segments_gaps = segments_gaps_flatten.reshape((-1, 2))

        gaps_lengths = np.linalg.norm(segments_gaps, axis=0)
        gaps_mask = gaps_lengths[gaps_lengths > gap_length]
        filtered_gaps = segments_gaps[gaps_mask]

        filtered_gaps_flatten = filtered_gaps.flatten()
        segments_flatten = np.insert(filtered_gaps_flatten, 0, self.segments[0, 0])
        segments_flatten = np.append(segments_flatten, self.segments[-1, -1])

        self.segments = segments_flatten.reshape((-1, 2))


    def combine_adjacent(self) -> None:
        """
        Description:
            Combine in place adjacent segments. E.g. segments [0, 199] and [200, 599] will be combined to [0, 599] segment.
        """
        for index in range(1, len(self.segments)):
            if self.segments[index - 1, 1] + 1 == self.segments[index, 0]:
                self.segments[index, 0] = self.segments[index - 1, 0]
                self.segments[index - 1] = -1

        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]


    def write(self, filepath: str) -> None:
        """
        Description:
            Write segments to file using numpy.write() function.

        :raises OSError: if ``filepath`` does not exist.
        """
        directory_name = os.path.dirname(filepath)
        if os.path.exists(directory_name):
            np.save(filepath, self.segments)
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
        return self.segments.size

    @property
    def shape(self) -> tuple:
        """
        Description:
            Returns segments shape.

        :return: segments shape
        """
        return self.segments.shape

    @property
    def lengths(self) -> np.ndarray:
        """
        Description:
            Returns segments lengths.

        :return: segments lengths
        """
        return np.linalg.norm(self.segments, axis=1)
