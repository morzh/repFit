from dataclasses import dataclass
from filters.persons_tracking.core.bounding_box import BoundingBox


@dataclass
class PersonVideoSegment:
    """
    Class containing information about video segment at which person's tracking is stable (using some tracking network).
    """
    def __init__(self, person_id):
        """
        Description:
            Class constructor.
        :param person_id: person's id (from tracking network)
        """
        self.id: int = person_id
        self.bounding_box: BoundingBox = BoundingBox()
        self.segment: tuple[int, int] = (-1, -1)
        # self.minimum_confidence = 1.0
        # self.maximum_confidence = 0.0

    def update(self, bounding_box: BoundingBox, frame_number: int) -> None:
        """
        Description:
            Update information about video segment at which person is considered to be presented.

        :param bounding_box: tracked bounding box of a person at frame_number
        :param frame_number: frame number
        :return: None
        """
        self.bounding_box.circumscribe(bounding_box)
        if self.segment[0] == -1:
            self.segment = (frame_number, -1)
        else:
            self.segment = (self.segment[0], frame_number)
