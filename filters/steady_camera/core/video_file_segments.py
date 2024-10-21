import copy

from utils.cv.video_properties import VideoProperties
from utils.cv.video_frames_segments import VideoFramesSegments

from typing import Self


class VideoFileSegments:
    """
    Description:
        Data storage class for video segments information.

    :ivar metadata: video file meta data;
    :ivar frames_segments: array of frames segments [[segment1_frame_start, segment1_frame_end], [segment2_frame_start, segment2_frame_end], ...]
    """

    __slots__ = ['video_properties', 'frames_segments']
    def __init__(self, video_properties: VideoProperties, segments: VideoFramesSegments):
        self.video_properties: video_properties
        self.frames_segments = segments

    def filter_by_time(self, time_period_threshold: float) -> None:
        """
        Description:
            Filter video segments by duration in place. If segment duration is less than time_period_threshold, it will be deleted.

        :param time_period_threshold: time threshold in seconds
        """
        self.frames_segments.filter_by_time(self.video_properties.fps, time_period_threshold)

    def complement(self) -> Self:
        r"""
        Description:
            Video segments complement set closure, where set is a  :math:`[0, number_frames - 1]` segment. Formula:

        :return:  video file segments complement
        """
        video_file_segments = copy.copy(self)
        video_file_segments.frames_segments = self.frames_segments.complement(self.video_properties.frames_number)
        return video_file_segments

    def whole_video_segments_check(self) -> bool:
        """
        Description:
            Checks if video_segments has only one segment with frame start equals zero and frame end equals frames number - 1

        :return: True if there is only one whole range video segment, False otherwise
        """
        shape_check = self.frames_segments.segments.shape[0] == 1
        zero_frame_check = self.frames_segments.segments[0, 0] == 0
        number_frames_check = self.frames_segments.segments[-1, -1] == self.video_properties.frames_number - 1
        return shape_check and zero_frame_check and number_frames_check
