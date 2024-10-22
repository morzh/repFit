import copy
import numpy as np
import os

from typing import Self


class Segments:
    """
    Description:
        Data storage class for video segments information.

    :ivar segments: array of frames segments [[segment1_frame_start, segment1_frame_end], [segment2_frame_start, segment2_frame_end], ...]
    """
    __slots__ = ['segments']
    def __init__(self, segments: np.ndarray):
        self.segments = segments

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

    def write(self, filepath: str):
        """
        Description:

        """
        directory_name = os.path.dirname(filepath)
        if os.path.exists(directory_name):
            np.save(filepath, self.segments)
        else:
            raise OSError('Filepath directory does not exist.')


    def combine_adjacent_segments(self) -> None:
        """
        Description:
            Combine adjacent segments. E.g. segments [0, 199] and [200, 599] will be combined to [0, 599] segment.
        """
        for index in range(1, len(self.segments)):
            if self.segments[index - 1, 1] + 1 == self.segments[index, 0]:
                self.segments[index, 0] = self.segments[index - 1, 0]
                self.segments[index - 1] = -1

        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]

    @property
    def size(self) -> int:
        return segment.size

    @property
    def shape(self) -> tuple:
        return self.segments.shape
