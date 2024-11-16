import numpy as np

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class AreaFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
        Filter person by mean area in pixels.

    :ivar area: area in pixels
    """

    def __init__(self, area):
        self.area = area

    def process(self, tracks: MultiplePersonsTracks):
        pass