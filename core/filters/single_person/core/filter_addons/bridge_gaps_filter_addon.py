from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class BridgeGapsFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter for bridging gaps between frames segments.
        If gap between neighbouring segments is less than the given threshold value, this segments will be fused into one.
        Threshold is in seconds.

    :ivar gap_threshold: segments gap threshold in seconds
    """
    def __init__(self, gap_threshold=5.0):
        self.gap_threshold = gap_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        video_fps = tracks.video_properties.fps
        for person in tracks.persons.values():
            if filter_full_body_person:
                person.full_body_data.bridge_gaps(video_fps)
            else:
                person.data.bridge_gaps(video_fps)
