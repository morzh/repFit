from abc import ABC

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class WholePersonFilterAddon(MultiPersonsFilterAddonBase, ABC):
    """
    Description:
    """
    def __init__(self):
        ...

    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
        """
