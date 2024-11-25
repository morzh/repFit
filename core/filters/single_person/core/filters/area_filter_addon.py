from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter each person track by mean person's bounding boxes area in pixels.

    :ivar area_threshold: threshold area in pixels
    """
    def __init__(self, area_threshold = 500):
        self.area_threshold = area_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Filter out each person's track, whose mean bounding box area less than the given threshold.
            Mean area is given in pixels.

        :param tracks: person's tracks.
        """
        persons_number = len(tracks.persons)
        for id_person in range(persons_number):
            current_mean_area = tracks.persons[id_person].mean_area()
            if current_mean_area < self.area_threshold:
                del tracks.persons[id_person]
