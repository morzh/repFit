from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.single_person_track import SinglePersonStatus


class WholePersonFilterAddon(MultiPersonsFilterAddonBase):
    """
    Description:
    """
    def __init__(self):
        ...

    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
        """
