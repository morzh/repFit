from abc import abstractmethod
import cv2.typing


class OcrBase:
    """
    Description:
        Base class for OCR text region mask detection. Class member 'alias' is used in a factory class as a key in dictionary.
    """
    alias: str

    def __init__(self, **kwargs):
        """
        Description:
            Create instance of a class with the given key word arguments.

        :return: __init__() should return None
        """
        ...

    @abstractmethod
    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        """
        Description:
            Get text regions mask for a given input image. Where mask value is 1, there should be a text region.

        :param image: input image
        :param output_resolution: resolution for output mask

        :return: image mask, whose values are in [0, 1] segment
        """
        ...

    @classmethod
    def create_instance(cls, **kwargs):
        """
        Description:
            Create instance of a class with the given kwargs.

        :return: class instance
        """
        class_kwargs = kwargs.get(cls.alias)
        return cls(**class_kwargs)
