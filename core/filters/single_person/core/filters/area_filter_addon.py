import numpy as np

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter person by mean area of person's bounding boxes in pixels.

    :ivar area: area in pixels
    """

    def __init__(self, area):
        self.area = area

    def process(self, tracks: MultiplePersonsTracks):
        """

        """
        persons_number = len(tracks.persons)
        for id_person in range(persons_number):
            current_mean_area = tracks.persons[id_person].mean_area()
            if current_mean_area < self.area:
                del tracks.persons[id_person]
