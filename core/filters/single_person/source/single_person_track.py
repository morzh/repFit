import numpy as np

from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.person_data import PersonData

from core.utils.cv.frames_segments import FramesSegments
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class SinglePersonTrack:
    """
    Description:
        Class containing information about video segment at which person's tracking is stable (using some tracking network).

    :ivar tracked_data: data, obtained from person's tracker
    :ivar data: person data
    :ivar full_body_data: fll body person data
    """
    def __init__(self):
        self.tracked_data = PersonTrackedData()
        self.data = PersonData()
        self.full_body_data = PersonData()


    def append(self, bounding_box: np.ndarray, frame_index: int, confidence: float, joints: np.ndarray | None = None) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_index: frame number
        :param confidence: tracked bounding box confidence
        :param joints: tracked keypoints
        """
        self.tracked_data.append(bounding_box, frame_index, confidence, bounding_box_mode=BoundingBoxes2DArray.XYXY, joints=joints)

    '''
    def calculate_full_body_person_segments(self, stride) -> FramesSegments:
        """
        Description:
            Calculate frames segments from tracked data and whole person index frames.

        :param stride: input video frames stride

        :return: whole person frame segments
        """
        frames_indices = self.full_body_data.frames_indices.values
        segments_bins = np.hstack((frames_indices.reshape(-1, 1), frames_indices.reshape(-1, 1) + stride))

        for index in range(segments_bins.shape[0] - 1):
            if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                segments_bins[index + 1, 0] = segments_bins[index, 0]
                segments_bins[index] = -1

        mask = segments_bins[:, 0] != -1
        segments = segments_bins[mask]
        segments[:, 1] += 1

        return FramesSegments(segments)
    '''


    # def filter_by_duration(self, fps: float, time_threshold: float = 5) -> None:
    #     """
    #     Description:
    #         Filter person's video segments by duration
    #
    #     :param fps: input video frames per second
    #     :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
    #     """
    #     if self.segments.size == 0: return
    #
    #     frames_threshold = round(fps * time_threshold)
    #     self.segments.filter_by_length(frames_threshold)


    # def bridge_gaps(self, fps: float, time_threshold: float = 5) -> None:
    #     """
    #     Description:
    #         If there is a gap between two adjacent frame segments, just fill it out.
    #         Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.
    #
    #     :param fps: input video frames per second
    #     :param time_threshold: time threshold of the gap in seconds
    #     """
    #     if len(self.segments) <= 1: return
    #     frames_gap_threshold = round(fps * time_threshold)
    #     self.segments.bridge_gaps(frames_gap_threshold)


    def mean_area(self) -> float:
        """
        Description:
            Calculates mean of all person's bounding boxes areas.

        :return: mean area of all person's bounding boxes.
        """
        return self.tracked_data.bounding_boxes.mean_area()


    def mean_height(self) -> float:
        """
        Description:
            Calculates mean height of all person's bounding boxes.

        :return: mean heights
        """
        return self.tracked_data.bounding_boxes.mean_height()


    # def mean_area_per_segment(self) -> np.ndarray:
    #     """
    #     Description:
    #         Calculates mean of bounding boxes areas of a person per frame segment.
    #
    #     :return: mean bounding boxes area per fame segment
    #     """
    #     if self.segments.size == 0: return np.array([])
    #
    #     indices_array = self.segments.frames_indices()
    #     mean_areas = np.empty(len(indices_array))
    #     for index, indices in enumerate(indices_array):
    #         mean_areas[index] = self.tracked_data.bounding_boxes.mean_area(indices)
    #
    #     return mean_areas


    def bounding_boxes_per_segment(self, segments: FramesSegments) -> BoundingBoxes2DArray:
        """
        Description:
            Calculate overall bounding box for each segment.

        :param segments: frames segments

        :return: segment's bounding boxes.
        """
        boxes = BoundingBoxes2DArray()
        for segment in segments:
            current_upper_bound_indices = np.argwhere(self.tracked_data.frames_indices < segment[1]).flatten()
            current_lower_bound_indices = np.argwhere(segment[0] <= self.tracked_data.frames_indices).flatten()
            current_indices = np.intersect1d(current_upper_bound_indices, current_lower_bound_indices)
            bounding_box = self.tracked_data.bounding_boxes.circumscribe(current_indices)
            bounding_box = bounding_box.astype(np.int32)
            boxes.append(bounding_box)
        return boxes


    def is_track_equals_video(self, video_properties: VideoProperties, frames_number: int) -> bool:
        """
        Description:
            Returns true, when two following conditions satisfied:
                1. Track has only one segment, equals to  [0, video_frames_number].
                2. Overall bounding box has is [0, 0, width-1, height-1]
            and returns False otherwise.

        :returns: True is track concise with video, False otherwise.
        """
        segments = self.full_body_data.frames_segments
        is_single_segment_equals_video_range = len(segments) == 1 and segments[0, 0] == 0 and segments[0, -1] == (frames_number - 1)
        bounding_box = self.tracked_data.bounding_boxes.circumscribe()
        is_single_bounding_box_matches_video_resolution = bounding_box[2:] == video_properties.resolution
        return is_single_segment_equals_video_range and is_single_bounding_box_matches_video_resolution


    def is_data_empty(self) -> bool:
        """

        """
        if not len(self.tracked_data.bounding_boxes) and not self.tracked_data.joints.shape[0] and not self.tracked_data.confidences.shape[0]:
            return True
        return False


    def are_segments_empty(self):
        """

        """
