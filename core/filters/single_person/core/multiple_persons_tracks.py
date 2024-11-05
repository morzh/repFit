import numpy as np
import torch

from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_box_2d import BoundingBox2D
from core.filters.single_person.core.single_person_track import SinglePersonTrack


class MultiplePersonsTracks:
    def __init__(self, video_properties: VideoProperties):
        """
        Description:

        """
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties


    def update(self, data: torch.Tensor, frame_number: int):
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            if current_person_id not in self.persons:
                # current_confidence = data[index, 5]
                self.persons[current_person_id] = SinglePersonTrack(current_person_id)
            bounding_box = BoundingBox2D(int(data[index, 0]), int(data[index, 1]), int(data[index, 2]), int(data[index, 3]))
            self.persons[current_person_id].update(bounding_box, frame_number)


    def filter_by_area_ratio(self, ratio: float = 4.0):
        """
        Description:
            Filter persons track which mean bounding box area is less than the biggest bounding box mean area .

        :param ratio: areas ratio
        """
        persons_number = len(self.persons)
        persons_areas = np.zeros(persons_number)
        for id_person in range(persons_number):
            persons_areas[id_person] = self.persons[id_person].mean_person_area()

        person_maximum_area = np.max(persons_areas)
        area_threshold = person_maximum_area / ratio
        small_persons_indices = np.argwhere(persons_areas < area_threshold)

        for key in small_persons_indices:
            del self.persons[key]


    def filter_by_area(self, area: int = 3500):
        """
        Description:
            Filter person by mean area in pixels.

        :param area: area in pixels
        """

    def filter_by_time(self, duration=5.0):
        """
        Description:

        :param duration: duration in seconds
        """
        for person in self.persons.values():
            person.filter_duration(self.video_properties.fps, duration)
