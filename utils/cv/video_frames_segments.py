import copy
import numpy as np
import os

from typing import Self

from oauthlib.uri_validate import segment


class VideoFramesSegments:
    """
    Description:

    """
    __slots__ = ['segments']
    def __init__(self, segments: np.ndarray):
        self.segments = segments

    def filter_by_time(self, video_fps: float, threshold: float) -> None:
        """
        Description:
            Filter video segments by duration in place.
            If segment duration is less than time_period_threshold, it will be deleted.

        :param video_fps: FPS of the video
        :param threshold: time threshold in seconds
        """
        for segment_index, segment in enumerate(self.segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / video_fps) < threshold:
                self.segments[segment_index] = np.array([-1, -1])
        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]

    def complement(self, frames_number: int) -> Self:
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
