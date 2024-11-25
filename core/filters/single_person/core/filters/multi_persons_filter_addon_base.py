from abc import ABC, abstractmethod

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


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
