from dataclasses import dataclass
import numpy as np

from core.utils.cv.segments import Segments
from core.utils.cv.segments_with_bounding_boxes import SegmentsWithBoundingBoxes
from core.utils.geometry.bounding_box_2d import BoundingBox2D
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray
import core.utils.geometry.bounding_box_2d_dyadic as bbox_bin_op



@dataclass
class SinglePersonTrack:
    """
    Description:
        Class containing information about video segment at which person's tracking is stable (using some tracking network).

    :ivar id:
    :ivar data:
    :ivar stride: frames stride
    :ivar stride: frames track stride value
    :ivar __previous_frame_number: previous frame index
    """

    def __init__(self, person_id, stride=1):
        """
        Description:
            Class constructor.

        :param person_id: person's id (from person tracking algorithm)
        """
        self.id: int = person_id
        self.data = SegmentsWithBoundingBoxes()
        self.stride = stride
        self.__previous_frame_number: int = -1

    def update(self, bounding_box: np.ndarray[], frame_number: int) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_number: frame number
        """
        if (frame_number - self.__previous_frame_number) == self.stride:
            self.data.append(frame_number, bounding_box, interpolate=True)
        else:
            self.data.append(frame_number, bounding_box, interpolate=False)


    def filter_duration(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            Filter person's video segments by duration

        :param fps: input video frames per second
        :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
        :return: None
        """
        for segment_index, segment in enumerate(self.data.segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / fps) < time_threshold:
                self.frames_segments[segment_index] = np.array([-1, -1])
        mask = self.frames_segments[:, 0] >= 0
        self.frames_segments = self.frames_segments[mask]

    def bridge_gaps(self, time_threshold: int = 5) -> None:
        """
        Description:
            If there is a gap between two adjacent frame segments, just fill it out.
            Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.

        :param time_threshold: time threshold of the gap in seconds
        :return: None
        """
        for segment_index, segment in enumerate(self.frames_segments.segments - 1):
            current_segments_gap = self.frames_segments[segment_index + 1][0] - self.frames_segments[segment_index][1]
            if current_segments_gap < time_threshold:
                self.frames_segments[segment_index + 1][0] = self.frames_segments[segment_index][0]
                self.frames_segments[segment_index] = -1

        mask = self.frames_segments[:, 0] >= 0
        self.frames_segments = self.frames_segments[mask]

    def mean_person_area(self) -> float:
        """
        Description:
            Calculates mean of all bounding boxes areas of a person.

        :return: mean area of all person's bounding boxes.
        """
        return np.mean(self.data.bounding_boxes.areas())
