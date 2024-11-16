from abc import ABC, abstractmethod

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class MultiPersonsFilterAddonBase(ABC):
    """
    Description:
        Base abstract filter class  MultiplePersonsTracks processing.
    """
    @abstractmethod
    def process(self, tracks: MultiplePersonsTracks) -> None:
        """
        Description:
            Abstract ``MultiplePersonsTracks`` filter method
            
        :param tracks: instance of ``MultiplePersonsTracks`` class.
        """