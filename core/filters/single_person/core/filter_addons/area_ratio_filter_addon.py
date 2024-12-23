import numpy as np

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AreaRatioFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is two-step filter. On the first step the biggest mean bounding box value calculated for each single person's track.
        On the second step, filter deletes tracks whose mean bounding box area is less than value, calculated at the first one.

    :ivar area_ratio_threshold: persons mean areas ratio threshold.
    """
    def __init__(self, area_ratio_threshold=4):
        self.area_ratio_threshold = area_ratio_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        """
        Description:
            Filter persons track which mean bounding box area is less than the biggest bounding box mean area .

        :param tracks: person's track
        :param filter_full_body_person: person's track
        """
        persons_areas = []
        persons_keys = []
        for person_id, person in tracks.persons.items():
            persons_areas.append(person.mean_area())
            persons_keys.append(person_id)

        persons_areas = np.array(persons_areas)
        person_maximum_area = np.max(persons_areas)
        area_threshold = person_maximum_area / self.area_ratio_threshold
        area_threshold_persons_indices = np.argwhere(persons_areas >= area_threshold)

        for key in area_threshold_persons_indices:
            current_frames_indices = tracks.persons[persons_keys[key[0]]].tracked_data.frames_indices
            if filter_full_body_person:
                tracks.persons[persons_keys[key[0]]].full_body_person_data.frames_indices = current_frames_indices
            else:
                tracks.persons[persons_keys[key[0]]].person_data.frames_indices = current_frames_indices

        # for key in area_threshold_persons_indices:
        #     del tracks.persons[persons_keys[key[0]]]
