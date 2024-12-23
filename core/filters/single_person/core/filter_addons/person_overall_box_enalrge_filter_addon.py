from abc import ABC

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class PersonOverallBoxEnlarge(MultiPersonsFilterAddonBase, ABC):
    """
    Description:
    """
    # def __init__(self):
    #     ...

    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person) -> None:
        """
        Description:
        """
