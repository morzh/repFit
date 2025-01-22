from __future__ import annotations
import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.source.frames_indices import FramesIndices


class PersonData:
    """
    Description:
        Class containing frames indices and respective segments.

    :ivar frames_segments: frames segments
    :ivar frames_indices: frames indices
    """
    def __init__(self):
        self.frames_segments = FramesSegments()
        self.frames_indices = FramesIndices()


    def __eq__(self, other):
        return self.frames_segments == other.frames_segments and self.frames_indices == other.frames_indices


    def insert(self, frames_indices: np.ndarray) -> None:
        self.frames_indices.insert(frames_indices)
        # self.frames_segments.update()


    def calculate_segments(self, stride=1) -> None:
        """
        Description:
            Calculate frames segments using  video  frames``stride`` value.

        :param stride: video frames stride
        """
        if len(self.frames_segments) > 0 or not len(self.frames_indices): return

        segments_bins = np.hstack((self.frames_indices.values.reshape(-1, 1), self.frames_indices.values.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = -1

        mask = segments_bins[:, 0] != -1
        segments = segments_bins[mask]
        segments[:, 1] += 1
        self.frames_segments = FramesSegments(segments)


    def clip_segments(self, segments: FramesSegments) -> FramesSegments:
        """
        Description:

        :param segments:
        """
        clipped_segments = FramesSegments()
        for segment in segments:
            current_segments = self.frames_segments.clip(segment[0], segment[1])
            clipped_segments.append(current_segments)
        return clipped_segments


    def filter_by_duration(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            Filter person's video segments by duration

        :param fps: input video frames per second
        :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
        """
        if self.frames_segments.size == 0: return
        frames_threshold = round(fps * time_threshold)
        self.frames_segments.filter_by_length(frames_threshold)


    def bridge_gaps(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            If there is a gap between two adjacent frame segments, just fill it out.
            Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.

        :param fps: input video frames per second
        :param time_threshold: time threshold of the gap in seconds
        """
        if len(self.frames_segments) <= 1: return
        frames_gap_threshold = round(fps * time_threshold)
        self.frames_segments.bridge_gaps(frames_gap_threshold)
#