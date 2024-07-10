import copy
from dataclasses import dataclass
from typing import Self
import numpy as np


@dataclass(slots=True)
class VideoSegments:
    """
    Description:
        Data storage class for video segments information.
    """
    video_filename: str
    video_width: int
    video_height: int
    video_fps: float
    frames_number: int
    segments: np.ndarray

    def filter_by_time_duration(self, time_period_threshold: float) -> None:
        """
        Description:
            Filter video segments by duration in place.
            If segment duration is less than time_period_threshold, it will be deleted.

        :param time_period_threshold: time threshold in seconds
        """
        for segment_index, segment in enumerate(self.segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / self.video_fps) < time_period_threshold:
                self.segments[segment_index] = np.array([-1, -1])
        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]

    def complement(self) -> Self:
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
        segments = np.append(segments, self.frames_number - 1)
        segments = segments.reshape(-1, 2)

        if segments[0, 0] == segments[0, 1]:
            segments = np.delete(segments, 0, axis=0)
        if segments[-1, 0] == segments[-1, 1]:
            segments = np.delete(segments, -1, axis=0)

        video_segments_complement = copy.copy(self)
        video_segments_complement.segments = segments

        return video_segments_complement

    def whole_video_segments_check(self) -> bool:
        """
        Description:
            Checks if video_segments has only one segment with frame start equals zero and frame end equals frames number - 1
        :return: True if there is only one whole range video segment, False otherwise
        """
        return (self.segments.shape[0] == 1 and
                self.segments[0, 0] == 0 and
                self.segments[-1, -1] == self.frames_number - 1)

    def write(self, filepath: str):
        np.save(filepath, self.segments)

    def combine_adjacent_segments(self) -> None:
        """
        Description:
            Combine adjacent segments. E.g. segments [0, 199] and [200, 599] will be combined to [0, 599] segment.
            TO_DO: Make this algorithm more efficient in terms of memory and speed
        """
        for index in range(1, len(self.segments)):
            if self.segments[index - 1][1] + 1 == self.segments[index][0]:
                self.segments[index - 1][1] = self.segments[index][1]
                self.segments[index][0] = self.segments[index - 1][0]

        self.segments = np.unique(self.segments, axis=0)
