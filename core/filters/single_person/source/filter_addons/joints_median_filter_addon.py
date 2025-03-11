import numpy as np
from ordered_set import OrderedSet
from scipy.ndimage import median_filter

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class JointsMedianFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Joints median temporal filter.
    """
    def __init__(self, sliding_half_window=2):
        self.sliding_half_window = sliding_half_window

    def process(self, tracks: MultiplePersonsTracks) -> None:
        if not len(tracks.persons): return
        filter_window_size = 2 * self.sliding_half_window + 1

        for person_track in tracks.persons.values():
            if not len(person_track.tracked_data) or not person_track.is_active: continue
            person_track.tracked_data.joints = median_filter(person_track.tracked_data.joints, size=filter_window_size, axes=0)