from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
# from core.filters.single_person.core.single_person_track import SinglePersonStatus


class BridgeGapsFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter for bridging gaps between frames segments.
        If gap between neighbouring segments is less than the given threshold  value, this segments will be fused into one.

    :ivar gap_threshold: segments gap threshold in seconds
    """
    def __init__(self, gap_threshold=5.0):
        self.gap_threshold = gap_threshold


    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Fill the gap between neighbouring segments within the given threshold.

        :param tracks: person's track
        """
        video_fps = tracks.video_properties.fps
        for person in tracks.persons.values():
            person.bridge_gaps(video_fps)
            # person.information.filters_applied.append('bridge_gap')
            # person.information.track_status = SinglePersonStatus.FILTERED
