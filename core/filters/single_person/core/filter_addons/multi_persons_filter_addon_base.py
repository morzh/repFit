# trick to avoid circular dependencies error. Here MultiplePersonsTracks import used only for type hinting
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks

from abc import ABC, abstractmethod


class MultiPersonsFilterAddonBase(ABC):
    """
    Description:
        Base abstract filter class for MultiplePersonsTracks processing.
    """

    @abstractmethod
    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Abstract method for filtering ``tracks`` using visitor pattern.
            
        :param tracks: tracks to filter.
        """
