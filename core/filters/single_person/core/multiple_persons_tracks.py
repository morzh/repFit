import torch

from core.filters.single_person.core.filters.multi_persons_filter_base import MultiPersonsFilterBase
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.core.single_person_track import SinglePersonTrack


class MultiplePersonsTracks:
    def __init__(self, video_properties: VideoProperties):
        """
        Description:

        """
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties


    def update(self, data: torch.Tensor, frame_number: int):
        """
        Description:
        """
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            if current_person_id not in self.persons:
                # current_confidence = data[index, 5]
                self.persons[current_person_id] = SinglePersonTrack(current_person_id)
            bounding_box = data[index].numpy()
            self.persons[current_person_id].update(bounding_box, frame_number)


    def apply_filter(self, filter_visitor: MultiPersonsFilterBase) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        """
        filter_visitor.process(self)
