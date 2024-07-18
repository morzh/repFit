import torch

from filters.persons_tracking.core.bounding_box import BoundingBox
from filters.persons_tracking.core.person_video_segment import PersonVideoSegment


class PersonsVideoSegments:
    def __init__(self, fps=30):
        self.persons: dict[int, PersonVideoSegment] = {}
        self.fps = fps

    def update(self, data: torch.Tensor, frame_number: int):
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            if current_person_id not in self.persons:
                # current_confidence = data[index, 5]
                self.persons[current_person_id] = PersonVideoSegment(current_person_id)
            bounding_box = BoundingBox(int(data[index, 0]),
                                       int(data[index, 1]),
                                       int(data[index, 2]),
                                       int(data[index, 3]))
            self.persons[current_person_id].update(bounding_box, frame_number)

    def filter_by_area(self, factor=3):
        ...

    def filter_by_duration(self, duration=5):
        for person in self.persons.values():
            person.filter_duration(self.fps, duration)
