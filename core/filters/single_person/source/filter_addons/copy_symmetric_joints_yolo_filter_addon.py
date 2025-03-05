import numpy as np
from ordered_set import OrderedSet

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks


class CopySymmetricJointsYoloFilterAddon(MultiPersonsFilterAddonBase):

    def process(self, tracks: MultiplePersonsTracks) -> None:
        pass