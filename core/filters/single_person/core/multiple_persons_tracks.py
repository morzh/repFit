import os
import pickle
import torch

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.core.single_person_track import SinglePersonTrack


class MultiplePersonsTracks:
    """
    Description:
        Persons data storage class.

    :ivar persons: person_id -> person data mapping
    :ivar video_properties:  video properties data
    """
    def __init__(self, video_properties: VideoProperties):
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties


    def update(self, data: torch.Tensor, frame_number: int) -> None:
        """
        Description:
            Update persons tracks data.

        :param data: person's data, obtained from AI model;
        :param frame_number: frame number.
        """
        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])
            current_confidence = float(data[index, 5])
            if current_person_id not in self.persons:
                self.persons[current_person_id] = SinglePersonTrack()
            bounding_box = data[index].numpy()
            self.persons[current_person_id].append(bounding_box, frame_number, confidence=current_confidence)


    def apply_filter(self, filter_visitor: MultiPersonsFilterAddonBase) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        """
        filter_visitor.process(self)

    def serialize(self, filepath: os.PathLike) -> None:
        """
        Description:
            Serialize class instance.

        :param filepath: filepath to write class instance to.
        """
        with open(filepath, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
