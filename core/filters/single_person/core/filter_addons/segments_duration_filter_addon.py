from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.single_person_track import SinglePersonStatus


class SegmentsDurationFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter every frame segment of person track in ``tracks`` by time.
        If duration of some frame segment will be less than ``duration_threshold``, it will be deleted.

    :ivar duration_threshold: duration threshold in seconds.
    """
    def __init__(self, duration=1.0):
        self.duration_threshold = duration


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Filter every frame segment of person track in ``tracks`` by time using threshold in seconds.

        :param tracks: person's tracks.
        """
        keys_to_delete = []
        for person_id, person in tracks.persons.items():
            person.filter_by_time(tracks.video_properties.fps, self.duration_threshold)
            if len(person.segments) == 0:
                keys_to_delete.append(person_id)

            person.information.filters_applied.append('segments_duration')
            person.information.track_status = SinglePersonStatus.FILTERED


        for key in keys_to_delete:
            tracks.persons.pop(key, None)