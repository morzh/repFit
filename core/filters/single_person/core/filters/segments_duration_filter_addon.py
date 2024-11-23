from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class SegmentsDurationFilterAddon(MultiPersonsFilterAddonBase):
    """

    """
    def __init__(self, duration=1.0):
        self.duration = duration

    def process(self, tracks: MultiplePersonsTracks) -> None:
        """

        """
        for person_track in tracks.persons.values():
            person_track.filter_by_time(tracks.video_properties.fps, self.duration)