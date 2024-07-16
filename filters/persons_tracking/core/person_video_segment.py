from dataclasses import dataclass
from filters.persons_tracking.core.bounding_box import BoundingBox


@dataclass
class PersonVideoSegment:
    id: int
    bounding_box: BoundingBox
    segment: list[int, int]

    def update(self, bounding_box: BoundingBox, frame: int):
        self.bounding_box.circumscribe(bounding_box)
        if not len(self.segment):
            self.segment.append(frame)
        elif len(self.segment) == 1:
            self.segment.append(frame)
        else:
            self.segment[1] = frame
