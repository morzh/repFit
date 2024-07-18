from dataclasses import dataclass
import numpy as np

from filters.persons_tracking.core.bounding_box import BoundingBox


@dataclass
class PersonVideoSegments:
    """
    Class containing information about video segment at which person's tracking is stable (using some tracking network).
    """

    def __init__(self, person_id):
        """
        Description:
            Class constructor.

        :param person_id: person's id (from person tracking algorithm)
        """
        self.id: int = person_id
        self.bounding_box: BoundingBox = BoundingBox()
        self.bounding_boxes_dimensions: np.ndarray = np.empty((0, 2))
        self.segments: list[tuple[int, int]] = []

    def update(self, bounding_box: BoundingBox, frame_number: int) -> None:
        """
        Description:
            Update information about video segments at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_number: frame number
        :return: None
        """
        self.bounding_box.circumscribe(bounding_box)
        if len(self.segments) == 0:
            self.segments.append((frame_number, frame_number))
        elif self.segments[-1][1] == (frame_number - 1):
            self.segments[-1] = (self.segments[-1][0], frame_number)
        else:
            self.segments.append((frame_number, frame_number))

    def filter_duration(self, fps: int, threshold: int = 5) -> None:
        for segment in self.segments:
            if (segment[1] - segment[0]) / fps < threshold:
                del segment

    def bridge_gaps(self, threshold=5) -> None:
        for index_segment, segment in enumerate(self.segments):
            current_segments_gap = self.segments[index_segment + 1][0] - self.segments[index_segment][1]
            if current_segments_gap < threshold:
                pass

    def mean_person_area(self) -> float:
        mean_bounding_boxes_dimensions = np.mean(self.bounding_boxes_dimensions, axis=0)
        return mean_bounding_boxes_dimensions[0] * mean_bounding_boxes_dimensions[1]
