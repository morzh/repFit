import numpy as np

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class AbsoluteAreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter out every person's track, whose mean bounding box area less than the given threshold.
        Mean area threshold is given in pixels. Filter taking into account only tracked data.
        So it should be used as one of the first filter in filtering pipeline.

    :ivar area_threshold: threshold area in pixels
    """
    def __init__(self, area_threshold=500):
        self.area_threshold = area_threshold


    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        if not len(tracks.persons):
            return

        if filter_full_body_person:
            self.filter_person_track_full_body_data(tracks)
        else:
            self.filter_person_track_data(tracks)


    def filter_person_track_data(self, tracks: MultiplePersonsTracks):
        for person_track in tracks.persons.values():
            if person_track.mean_area() < self.area_threshold:
                person_track.data.frames_indices = np.empty(0, )
            elif not len(person_track.data.frames_indices):
                person_track.data.frames_indices = person_track.tracked_data.frames_indices


    def filter_person_track_full_body_data(self, tracks: MultiplePersonsTracks):
        for person_id, person_track in tracks.persons.items():
            if person_track.mean_area() < self.area_threshold:
                person_track.full_body_data.frames_indices = np.empty(0, )
            elif not len(person_track.data.frames_indices):
                person_track.full_body_data.frames_indices = person_track.tracked_data.frames_indices
