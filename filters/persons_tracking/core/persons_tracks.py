import numpy as np
import torch

from utils.geometry.bounding_box_2d import BoundingBox2D
from filters.persons_tracking.core.person_id_track import PersonIdTrack


class PersonsTracks:
    def __init__(self, fps=30):
        """

        """
        self.fps = fps
        self.persons: dict[int, PersonIdTrack] = {}

    def update(self, data: torch.Tensor, frame_number: int):
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            if current_person_id not in self.persons:
                # current_confidence = data[index, 5]
                self.persons[current_person_id] = PersonIdTrack(current_person_id)
            bounding_box = BoundingBox2D(int(data[index, 0]), int(data[index, 1]), int(data[index, 2]), int(data[index, 3]))
            self.persons[current_person_id].update(bounding_box, frame_number)

    def filter_by_area(self, factor=3):
        persons_number = len(self.persons)
        persons_areas = np.zeros(persons_number)
        for id_person in range(persons_number):
            persons_areas[id_person] = self.persons[id_person].mean_person_area()

        person_maximum_area = np.max(persons_areas)
        area_threshold = person_maximum_area / factor
        small_persons_indices = np.argwhere(persons_areas < area_threshold)

        for key in small_persons_indices:
            del self.persons[key]

    def filter_by_duration(self, duration=5):
        for person in self.persons.values():
            person.filter_duration(self.fps, duration)
