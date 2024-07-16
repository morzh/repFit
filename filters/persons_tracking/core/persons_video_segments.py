import torch
from filters.persons_tracking.core.person_video_segment import PersonVideoSegment


class PersonsVideoSegments:
    def __init__(self):
        self.persons: dict[int, PersonVideoSegment] = {}

    def update(self, data: torch.Tensor):
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            self.persons[current_person_id] = PersonVideoSegment(current_person_id)
