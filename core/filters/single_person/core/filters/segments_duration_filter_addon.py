from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class SegmentsDurationFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:

    :ivar duration_threshold: duration threshold in seconds.
    """
    def __init__(self, duration=1.0):
        self.duration_threshold = duration


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:

        :param tracks: person's tracks.
        """
        for person_track in tracks.persons.values():
            person_track.filter_by_time(tracks.video_properties.fps, self.duration_threshold)