from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class BridgeGapsFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter for bridging gaps between frames segments.
        If gap between neighbouring segments is less than the given threshold value, this segments will be fused into one.

    :ivar gap_threshold: segments gap threshold in seconds;
    :ivar filter_full_body_person: if True filter full body data, just body data otherwise.
    """
    def __init__(self, **parameters):
        self.gap_threshold = parameters.get('gap_threshold', 5.0)
        self.filter_full_body_person = parameters.get('filter_full_body_person', True)


    def process(self, tracks: MultiplePersonsTracks) -> None:
        video_fps = tracks.video_properties.fps
        for person_track in tracks.persons.values():
            if not len(person_track.tracked_data) or not person_track.is_active: continue

            current_data_reference = person_track.full_body_data if self.filter_full_body_person else person_track.partial_body_data
            current_data_reference.calculate_segments(tracks.frames_stride)
            current_data_reference.bridge_gaps(video_fps, self.gap_threshold)
