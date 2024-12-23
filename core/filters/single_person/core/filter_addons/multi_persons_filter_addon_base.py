from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Visitor pattern is a cycling dependencies beast. To avoid circular dependencies error, import MultiplePersonsTracks only for type checking.
    from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks

from abc import ABC, abstractmethod


class MultiPersonsFilterAddonBase(ABC):
    """
    Description:
        Base abstract filter class for MultiplePersonsTracks processing.
    """

    @abstractmethod
    def process(self, tracks: MultiplePersonsTracks, filter_full_body_person=False) -> None:
        """
        Description:
            Abstract method for filtering ``tracks`` using visitor pattern.
            
        :param tracks: tracks to filter.
        :param filter_full_body_person: if True apply filter to full body person segments. If False apply filter to persons.
        """
