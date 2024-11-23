from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class BridgeGapsFilterAddon(MultiPersonsFilterAddonBase):
    """

    """
    def __init__(self, gap_threshold=5.0):
        self.threshold = gap_threshold

    def process(self, tracks: MultiplePersonsTracks) -> None:
        """

        """
        persons_number = len(tracks.persons)
        video_fps = tracks.video_properties.fps
        for id_person in range(persons_number):
            tracks.persons[id_person].bridge_gaps(video_fps)
