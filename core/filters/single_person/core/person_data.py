from __future__ import annotations
import numpy as np

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.frames_indices import FramesIndices


class PersonData:
    """

    """
    def __init__(self):
        self.frames_segments = FramesSegments()
        self.frames_indices = FramesIndices()


    def calculate_segments(self, stride=1) -> None:
        """
        Description:
            Calculate frames segments using ``stride`` value.

        :param stride: frames stride

        :return: frame segments
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
