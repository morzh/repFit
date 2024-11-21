import numpy as np

from core.utils.cv.video_properties import VideoProperties
from core.utils.cv.frames_segments import FramesSegments


class VideoFileSegments:
    """
    Description:
        Data storage class for video segments information.

    :ivar segments: video segments;
    :ivar video_properties: video file properties;
    :ivar frames_number: exact frames number;
    """
    __slots__ = ['segments', 'video_properties', 'frames_number']
    def __init__(self, segments: np.ndarray, video_properties: VideoProperties, frames_number: int):
        self.segments = FramesSegments(segments)
        self.video_properties = video_properties
        self.frames_number = frames_number


    def filter_by_time(self, time_threshold: float) -> None:
        """
        Description:
            Filter video segments by duration in place. If segment duration is less than time_period_threshold, it will be deleted.

        :param time_threshold: time threshold in seconds
        """
        frames_threshold = round(time_threshold * self.video_properties.fps)
        self.segments.filter_by_length(frames_threshold)


    def complement(self) -> None:
        """
        Description:
            Video segments complement set closure, where set is a  :math:`[0, number_frames - 1]` segment. Formula:

        :return: video file segments complement
        """
        self.segments.complement(0, self.frames_number)


    def is_whole_video_single_segment(self) -> bool:
        """
        Description:
            Checks if video_segments has only one segment with frame start equals zero and frame end equals frames number - 1

        :return: True if there is only one whole range video segment, False otherwise
        """
        is_single_segment = len(self.segments) == 1
        is_whole_video_range = (self.segments[0, 0] == 0) and (self.segments[0, -1] == (self.frames_number - 1))
        return is_single_segment and is_whole_video_range
