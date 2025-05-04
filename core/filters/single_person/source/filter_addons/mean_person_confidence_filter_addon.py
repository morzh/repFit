import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class MeanPersonConfidenceFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:

        Filter is working  only on tracked data. So it should be used as one of the first filter add-ons in filtering chain.

    :ivar mean_confidence_threshold: mean person confidence threshold.
    """
    def __init__(self, mean_confidence_threshold=0.5):
        self.mean_confidence_threshold = mean_confidence_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        if len(tracks.persons) < 2: return

        for person_id, person in tracks.persons.items():
            if not person.is_active: continue
            if np.mean(person.tracked_data.confidences) < self.mean_confidence_threshold:
                person.is_active = False
