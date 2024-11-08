from abc import ABC, abstractmethod

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks


class MultiPersonsFilterAddonBase(ABC):
    """
    Description:
        Base abstract filter class  MultiplePersonsTracks processing.
    """
    @abstractmethod
    def process(self, multi_persons_track: MultiplePersonsTracks) -> None:
        """
        Description:
            Abstract filter method
        """