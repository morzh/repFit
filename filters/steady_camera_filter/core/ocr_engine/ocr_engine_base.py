from abc import ABC, abstractmethod


class OcrEngineBase(ABC):

    @abstractmethod
    def pixel_mask(self, image, output_resolution):
        pass
