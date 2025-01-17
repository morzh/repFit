import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class AreaRatioFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is two-step filter.
        1. On the first step the biggest mean bounding box value calculated for each single person's track.
        2. On the second step, person's track data deleted, if mean bounding box area is less than value, calculated at first step.

        Filter taking into account only tracked data. So it should be used as one of the first filter in filtering pipeline.

    :ivar area_ratio_threshold: persons mean areas ratio threshold.
    """
    def __init__(self, area_ratio_threshold=4):
        self.area_ratio_threshold = area_ratio_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        if not len(tracks.persons):
            return

        persons_areas = []
        persons_keys = []

        for person_id, person in tracks.persons.items():
            persons_areas.append(person.mean_area())
            persons_keys.append(person_id)

        persons_areas = np.array(persons_areas)
        person_maximum_area = np.max(persons_areas)
        area_threshold = person_maximum_area / self.area_ratio_threshold
        persons_area_threshold_indices = np.argwhere(persons_areas < area_threshold)

        for key in persons_area_threshold_indices:
            if filter_full_body_person:
                tracks.persons[persons_keys[key[0]]].full_body_data.frames_indices = np.empty(0, )
            else:
                tracks.persons[persons_keys[key[0]]].data.frames_indices = np.empty(0, )

