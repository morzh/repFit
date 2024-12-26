from __future__ import annotations
import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.frames_indices import FramesIndices


class PersonData:
    """
    Description:

    :ivar frames_segments:
    :ivar frames_indices:
    """
    def __init__(self):
        self.frames_segments = FramesSegments()
        self.frames_indices = FramesIndices()


    def calculate_segments(self, stride=1) -> None:
        """
        Description:
            Calculate frames segments using  video  frames``stride`` value.

        :param stride: video frames stride
        """
        segments_bins = np.hstack((self.frames_indices.values.reshape(-1, 1), self.frames_indices.values.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = -1

        mask = segments_bins[:, 0] != -1
        segments = segments_bins[mask]
        segments[:, 1] += 1

        self.frames_segments.values = segments


    def clip_segments(self, segments: FramesSegments) -> FramesSegments:
        clipped_segments = FramesSegments()
        for segment in segments:
            current_segments = self.frames_segments.clip(segment[0], segment[1])
            clipped_segments.append(current_segments)

        return clipped_segments