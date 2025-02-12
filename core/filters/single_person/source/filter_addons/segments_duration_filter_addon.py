from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class SegmentsDurationFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter every frame's segment of a person track in ``tracks`` by time.
        If duration of some frame segment will be less than ``duration_threshold``, it will be deleted.

    :ivar duration_threshold: duration threshold in seconds.
    """
    def __init__(self, **parameters):
        self.duration_threshold = parameters.get('duration', 1.0)
        self.filter_full_body_person = parameters.get('filter_full_body_person', False)


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        for person_id, person_track in tracks.persons.items():
            if not len(person_track.tracked_data) or not person_track.is_active: continue

            current_data_reference = person_track.full_body_data if filter_full_body_person else person_track.body_data
            current_data_reference.calculate_segments(tracks.frames_stride)
            current_data_reference.filter_by_duration(tracks.video_properties.fps, self.duration_threshold)
