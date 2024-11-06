from dataclasses import dataclass
import numpy as np

from core.utils.cv.segments import Segments
from core.utils.geometry.bounding_box_2d import BoundingBox2D
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray
import core.utils.geometry.bounding_box_2d_dyadic as bbox_bin_op
from core.utils.geometry.indexed_bounding_boxes import IndexedBoundingBoxes


class SinglePersonTrack:
    """
    Description:
        Class containing information about video segment at which person's tracking is stable (using some tracking network).

    :ivar id:
    :ivar data:
    :ivar frame_segments:
    """

    def __init__(self, person_id, stride=1):
        """
        Description:
            Class constructor.

        :param person_id: person's id (from person tracking algorithm)
        """
        self.id: int = person_id
        self.data = IndexedBoundingBoxes()
        self.frame_segments = Segments()

    def update(self, bounding_box: np.ndarray, frame_number: int) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_number: frame number
        """
        self.data.append(bounding_box, frame_number)


    def filter_duration(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            Filter person's video segments by duration

        :param fps: input video frames per second
        :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
        """
        frames_threshold = round(fps * time_threshold)
        self.frame_segments.filter_by_length(frames_threshold)


    def bridge_gaps(self, fps: float, time_threshold: float = 5) -> None:
        """
        Description:
            If there is a gap between two adjacent frame segments, just fill it out.
            Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.

        :param fps: input video frames per second
        :param time_threshold: time threshold of the gap in seconds
        """
        frames_gap_threshold = round(fps * time_threshold)
        self.frame_segments.bridge_gaps(frames_gap_threshold)

    def mean_person_area(self) -> float:
        """
        Description:
            Calculates mean of all bounding boxes areas of a person.

        :return: mean area of all person's bounding boxes.
        """
        return self.data.mean_area()
