import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class AreaRatioFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        This is two-step filter.
        1. On the first step the biggest mean bounding box value calculated for each single person's track.
        2. On the second step, person's track data deleted, if mean bounding box area is less than value, calculated at first step.

        Filter is working  only on tracked data. So it should be used as one of the first filter add-ons in filtering chain.

    :ivar area_ratio_threshold: persons mean areas ratio threshold.
    """
    def __init__(self, area_ratio_threshold=4, mean_confidence_threshold=0.5):
        self.area_ratio_threshold = area_ratio_threshold
        self.mean_confidence_threshold = mean_confidence_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        if len(tracks.persons) < 2: return

        persons_areas = []
        persons_ids = []

        for person_id, person in tracks.persons.items():
            if not person.is_active: continue
            if np.mean(person.tracked_data.confidences) < self.mean_confidence_threshold: continue
            persons_areas.append(person.tracked_data.bounding_boxes.mean_area())
            persons_ids.append(person_id)

        persons_ids = np.array(persons_ids)
        persons_areas = np.array(persons_areas)
        person_maximum_area = np.max(persons_areas)
        absolute_area_threshold = person_maximum_area / self.area_ratio_threshold

        persons_mask = persons_areas < absolute_area_threshold
        inactive_tracks = persons_ids[persons_mask]

        for track_id in inactive_tracks:
            tracks.persons[track_id].is_active = False

