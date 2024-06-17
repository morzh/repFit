from abc import ABC, abstractmethod
from cv2 import typing


class OcrBase(ABC):
    @abstractmethod
    def pixel_mask(self, image, output_resolution) -> typing.MatLike:
        pass
