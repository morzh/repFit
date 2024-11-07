from core.filters.single_person.core.filters.multi_persons_filter_base import MultiPersonsFilterBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class SegmentsDurationFilter(MultiPersonsFilterBase):

    def __init__(self, duration=1):
        self.duration = duration

    def process(self, multi_persons_track: MultiplePersonsTracks):
        for person in multi_persons_track.persons.values():
            person.filter_by_time(multi_persons_track.video_properties.fps, self.duration)