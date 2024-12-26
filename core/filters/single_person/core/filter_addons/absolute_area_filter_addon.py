import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AbsoluteAreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out each person's track, whose mean bounding box area less than the given threshold.
        Mean area threshold is given in pixels.

    :ivar area_threshold: threshold area in pixels
    """
    def __init__(self, area_threshold=500):
        self.area_threshold = area_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        for person_id, person_track in tracks.persons.items():
            if person_track.mean_area() < self.area_threshold:
                if filter_full_body_person:
                    person_track.full_body_data.frames_indices = np.empty(0, )
                else:
                    person_track.data.frames_indices = np.empty(0, )
