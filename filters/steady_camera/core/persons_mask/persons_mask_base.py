from abc import ABC, abstractmethod
import cv2.typing


class PersonsMaskBase:
    """
    Description:
        Base class for persons detection and masking. Class member 'alias' is used in factory class as a key in dictionary.
    """
    alias: str

    def __init__(self, **kwargs):
        """
        Description:
            Create instance of a class with the given key word arguments.
        """
        ...

    @abstractmethod
    def pixel_mask(self, image: cv2.typing.MatLike, output_resolution: tuple[int, int]) -> cv2.typing.MatLike:
        """
        Description:
            Get all persons mask for a given input image. Where mask value is 1, there should be a some person.

        :param image: input image
        :param output_resolution: resolution for output mask

        :return: image mask, whose values are in [0, 1] segment
        """
        ...

    @classmethod
    def create_instance(cls, **kwargs):
        """
        Description:
            Create instance of a class with the given key word arguments.

        :return: class instance
        """
        class_kwargs = kwargs.get(cls.alias)
        return cls(**class_kwargs)
