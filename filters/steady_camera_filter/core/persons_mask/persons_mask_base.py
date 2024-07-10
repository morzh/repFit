from abc import ABC, abstractmethod
import cv2.typing


class PersonsMaskBase:
    @abstractmethod
    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        """
        Description:
            Get all persons mask for a given input image. Where mask value is 1, there should be a some person.
        @image: input image
        @output_resolution: resolution for output mask
        @return: image mask, whose values are in [0, 1] segment
        """
        pass
