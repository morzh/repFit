from abc import ABC, abstractmethod

import cv2.typing


class OcrBase(ABC):
    @abstractmethod
    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        """
        Description:
            Get text regions mask for a given input image. Where mask value is 1, there should be a text region.
        @image: input image
        @output_resolution: resolution for output mask
        @return: image mask, whose values are in [0, 1] segment
        """
        pass
