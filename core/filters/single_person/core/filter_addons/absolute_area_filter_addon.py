from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
# from core.filters.single_person.core.single_person_track import SinglePersonStatus


class AbsoluteAreaFilterAddon(MultiPersonsFilterAddonBase):
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
        keys_to_delete = []
        for person_id, person in tracks.persons.items():
            current_mean_area = person.mean_area()
            if current_mean_area < self.area_threshold:
                keys_to_delete.append(person_id)

            # person.information.filters_applied.append('absolute_area')
            # person.information.track_status = SinglePersonStatus.FILTERED


        for key in keys_to_delete:
            tracks.persons.pop(key, None)
