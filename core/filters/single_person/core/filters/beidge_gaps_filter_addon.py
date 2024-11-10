from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class BridgeGapsFilterAddon(MultiPersonsFilterAddonBase):
    """

    """
    def __init__(self, maximum_gap_time=5.0):
        ...

    def process(self, multi_persons_track: MultiplePersonsTracks) -> None:
        pass