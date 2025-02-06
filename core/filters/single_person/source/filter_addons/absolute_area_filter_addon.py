import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class AbsoluteAreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out every person's track, whose mean bounding box area of tracked data is less than the given threshold.
        Mean area threshold is given in pixels. Filter is working only on tracked data.
        So it should be used as one of the first filter add-on in filtering chain.

    :ivar area_threshold: threshold area in pixels
    """
    def __init__(self, area_threshold=500):
        self.area_threshold = area_threshold


    def process(self, tracks: MultiplePersonsTracks, **kwargs) -> None:
        if not len(tracks.persons):
            return

        keys_to_delete = []
        for person_id, person in tracks.persons.items():
            current_person_mean_area = person.mean_area()
            if current_person_mean_area < self.area_threshold:
                keys_to_delete.append(person_id)

        for key in keys_to_delete:
            tracks.persons.pop(key, None)
