import numpy as np

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AreaRatioFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter persons track which mean bounding box area is less than the biggest bounding box mean area .
    """

    def __init__(self, **parameters):
        self.ratio = parameters.get('ratio', 4)

    def process(self, tracks: MultiplePersonsTracks):
        """
        Description:
            Filter persons track which mean bounding box area is less than the biggest bounding box mean area .

        :param tracks: areas ratio
        """
        persons_number = len(tracks.persons)
        persons_areas = np.zeros(persons_number)
        for id_person in range(persons_number):
            persons_areas[id_person] = tracks.persons[id_person].mean_area()

        person_maximum_area = np.max(persons_areas)
        area_threshold = person_maximum_area / self.ratio
        small_persons_indices = np.argwhere(persons_areas < area_threshold)

        for key in small_persons_indices:
            del tracks.persons[key]