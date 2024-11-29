import numpy as np
from enum import Enum

from core.utils.cv.frames_segments import FramesSegments
from core.filters.single_person.core.person_tracking_data import PersonTrackingData
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


class SinglePersonStatus(Enum):
    NOT_FILTERED = 0
    FILTERED = 1
    READY_TO_WRITE = 2


class SinglePersonInformation:
    filters_applied: list = []
    track_status: SinglePersonStatus = SinglePersonStatus.NOT_FILTERED


class SinglePersonTrack:
    """
    Description:
        Class containing information about video segment at which person's tracking is stable (using some tracking network).

    :ivar data: data, obtained from person's tracker
    :ivar segments: video frame segments
    """

    def __init__(self):
        self.data = PersonTrackingData()
        self.segments = FramesSegments()
        self.information = SinglePersonInformation()


    def append(self, bounding_box: np.ndarray, frame_index: int, confidence: float) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_index: frame number
        :param confidence: bounding box confidence
        """
        self.data.append(bounding_box, frame_index, confidence, bounding_box_mode=BoundingBoxes2DArray.XYXY)


    def calculate_segments(self, stride):
        self.segments = self.data.calculate_segments(stride)


    def filter_by_time(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            Filter person's video segments by duration

        :param fps: input video frames per second
        :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
        """
        if self.segments.size == 0: return

        frames_threshold = round(fps * time_threshold)
        self.segments.filter_by_length(frames_threshold)


    def bridge_gaps(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            If there is a gap between two adjacent frame segments, just fill it out.
            Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.

        :param fps: input video frames per second
        :param time_threshold: time threshold of the gap in seconds
        """
        if len(self.segments) <= 1: return
        frames_gap_threshold = round(fps * time_threshold)
        self.segments.bridge_gaps(frames_gap_threshold)


    def mean_area(self) -> float:
        """
        Description:
            Calculates mean of all bounding boxes areas of a person.

        :return: mean area of all person's bounding boxes.
        """
        return self.data.bounding_boxes.mean_area()


    def mean_area_per_segment(self) -> np.ndarray:
        """
        Description:
            Calculates mean of bounding boxes areas of a person per frame segment.

        :return: mean bounding boxes area per fame segment
        """
        if self.segments.size == 0: return np.array([])

        indices_array = self.segments.as_frames_indices()
        mean_areas = np.empty(len(indices_array))
        for index, indices in enumerate(indices_array):
            mean_areas[index] = self.data.bounding_boxes.mean_area(indices)

        return mean_areas


    def is_track_equals_video(self, video_properties: VideoProperties, frames_number) -> bool:
        """
        Description:
            Returns true, when two following conditions satisfied:
                1. Track has only one segment, equals to  [0, video_frames_number].
                2. Overall bounding box has is [0, 0, width-1, height-1]
            and returns False otherwise.

        :returns: True is track concise with video, False otherwise.
        """
        is_single_segment_equals_video_range = len(self.segments) == 1 and self.segments[0, 0] == 0 and self.segments[0, -1] == (frames_number - 1)
        bounding_box = self.data.bounding_boxes.circumscribe()
        is_single_bounding_box_matches_video_resolution = bounding_box[2:] == video_properties.resolution
        return is_single_segment_equals_video_range and is_single_bounding_box_matches_video_resolution


    def bounding_boxes_per_segment(self) -> BoundingBoxes2DArray:
        """
        Description:
            Calculate overall bounding box for each segment.

        :return: segment's bounding boxes.
        """
        boxes = BoundingBoxes2DArray()
        for segment in self.segments:
            segment_indices = segment.as_frames_indices()
            bounding_box = self.data.bounding_boxes.circumscribe(segment_indices)
            boxes.append(bounding_box)
        return boxes
