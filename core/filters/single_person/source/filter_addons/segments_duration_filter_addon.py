from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class SegmentsDurationFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter every frames segment of a person track in ``tracks`` by time.
        If duration of some frame segment will be less than ``duration_threshold``, it will be deleted.

    :ivar duration_threshold: duration threshold in seconds.
    """
    def __init__(self, duration=1.0):
        self.duration_threshold = duration


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        for person_id, person_track in tracks.persons.items():
            if filter_full_body_person:
                person_track.full_body_data.calculate_segments(tracks.frames_stride)
                person_track.full_body_data.filter_by_duration(tracks.video_properties.fps, self.duration_threshold)
            else:
                person_track.data.calculate_segments(tracks.frames_stride)
                person_track.data.filter_by_duration(tracks.video_properties.fps, self.duration_threshold)
