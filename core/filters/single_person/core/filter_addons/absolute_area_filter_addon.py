from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AbsoluteAreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter each person track by mean person's bounding boxes area in pixels.

    :ivar area_threshold: threshold area in pixels
    """
    def __init__(self, area_threshold = 500):
        self.area_threshold = area_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        """
        Description:
            Filter out each person's track, whose mean bounding box area less than the given threshold.
            Mean area is given in pixels.

        :param tracks: person's tracks.
        :param filter_full_body_person: filter full body person.
        """
        for person_id, person in tracks.persons.items():
            if person.mean_area() >= self.area_threshold:
                current_frames_indices = person.tracked_data.frames_indices
                if filter_full_body_person:
                    person.full_body_person_data.frames_indices = current_frames_indices
                else:
                    person.person_data.frames_indices = current_frames_indices
