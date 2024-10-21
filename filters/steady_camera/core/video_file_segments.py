import copy

import numpy as np

from utils.cv.video_properties import VideoProperties
from utils.cv.video_frames_segments import VideoFramesSegments

from typing import Self


class VideoFileSegments(VideoFramesSegments):
    """
    Description:
        Data storage class for video segments information.

    :ivar segments: array of frames segments [[segment1_frame_start, segment1_frame_end], [segment2_frame_start, segment2_frame_end], ...]
    :ivar video_properties: video file properties;
    """

    __slots__ = ['video_properties']
    def __init__(self, segments: np.ndarray, video_properties: VideoProperties):
        super().__init__(segments)
        self.video_properties: video_properties
        # self.frames_segments = segments

    def filter_by_time(self, time_period_threshold: float, *args, **kwargs) -> None:
        """
        Description:
            Filter video segments by duration in place. If segment duration is less than time_period_threshold, it will be deleted.

        :param time_period_threshold: time threshold in seconds
        """
        super().filter_by_time(time_period_threshold, self.video_properties.fps)

    def complement(self, *args, **kwargs) -> Self:
        r"""
        Description:
            Video segments complement set closure, where set is a  :math:`[0, number_frames - 1]` segment. Formula:

        :return:  video file segments complement
        """
        video_file_segments = copy.copy(self)
        video_file_segments.segments = super().complement(self.video_properties.frames_number)
        return video_file_segments

    def whole_video_segments_check(self) -> bool:
        """
        Description:
            Checks if video_segments has only one segment with frame start equals zero and frame end equals frames number - 1

        :return: True if there is only one whole range video segment, False otherwise
        """
        shape_check = self.segments.shape[0] == 1
        zero_frame_check = self.segments[0, 0] == 0
        number_frames_check = self.segments[-1, -1] == self.video_properties.frames_number - 1
        return shape_check and zero_frame_check and number_frames_check
