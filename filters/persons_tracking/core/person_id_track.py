from dataclasses import dataclass
import numpy as np

from utils.geometry.bounding_box_2d import BoundingBox2D


@dataclass
class PersonIdTrack:
    """
    Description:
        Class containing information about video segment at which person's tracking is stable (using some tracking network).

    :ivar bounding_box: Bounding box containing tracked person with certain id across all frames
    :ivar bounding_boxes_areas: areas of tracked person's bounding boxes at each frame person was detected.
    :ivar segments: frame segments at which person with certain ID was tracked
    """

    def __init__(self, person_id):
        """
        Description:
            Class constructor.

        :param person_id: person's id (from person tracking algorithm)
        """
        self.id: int = person_id
        self.bounding_box: BoundingBox2D = BoundingBox2D()
        self.bounding_boxes_areas: np.ndarray = np.empty(0)
        self.segments: np.ndarray = np.empty((0, 2), dtype=np.int32)

    def update(self, bounding_box: BoundingBox2D, frame_number: int) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_number: frame number
        :return: None
        """
        self.bounding_box.circumscribeInPlace(bounding_box)
        self.bounding_boxes_areas = np.append(self.bounding_boxes_areas, bounding_box.area)
        if self.segments.shape[0] == 0:
            self.segments = np.vstack((self.segments, np.array([frame_number, frame_number])))
        elif self.segments[-1, 1] == (frame_number - 1):
            self.segments[-1, 1] = frame_number
        else:
            self.segments = np.vstack((self.segments, np.array([frame_number, frame_number])))

    def filter_duration(self, fps: int, time_threshold: int = 5) -> None:
        """
        Description:
            Filter person's video segments by duration

        :param fps: input video frames per second
        :param time_threshold: time threshold in seconds, if segment's  duration is less the threshold it will be deleted
        :return: None
        """
        for segment_index, segment in enumerate(self.segments):
            segment_length = segment[1] - segment[0]
            if (segment_length / fps) < time_threshold:
                self.segments[segment_index] = np.array([-1, -1])
        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]

    def bridge_gaps(self, time_threshold: int = 5) -> None:
        """
        Description:
            If there is a gap between two adjacent frame segments, just fill it out.
            Two given segments [t1, t2] [t3, t4] will be combined in to one [t1, t4] segment if a gap [t2, t3] less than a threshold.

        :param time_threshold: time threshold of the gap in seconds
        :return: None
        """
        for segment_index, segment in enumerate(self.segments - 1):
            current_segments_gap = self.segments[segment_index + 1][0] - self.segments[segment_index][1]
            if current_segments_gap < time_threshold:
                self.segments[segment_index + 1][0] = self.segments[segment_index][0]
                self.segments[segment_index] = -1

        mask = self.segments[:, 0] >= 0
        self.segments = self.segments[mask]

    def mean_person_area(self) -> float:
        """
        Description:
            Calculates mean of all bounding boxes areas of a person.

        :return: mean area of all person's bounding boxes.
        """
        return np.mean(self.bounding_boxes_areas)
