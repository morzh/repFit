import typing
from loguru import logger


class Factory:
    """
    Description:
        This class serves as a factory in a factory pattern.
    """
    def __init__(self):
        """
        Description:
            Initialize empty _builders dictionary.

        :return: __init__() should return None
        """
        self._builders = {}

    def register_builder(self, key: str, builder: typing.Callable) -> None:
        """
        Description:
            Register factory subject builder. Builder creates class, initialized with parameters.

        :param key: class alias key
        :param builder: subject class builder
        """
        self._builders[key] = builder

    def create(self, key: str, **kwargs) -> typing.Callable:
        r"""
        Description:
            Create initialized subject with the given key word parameters

        :param key: subject class key in builders dictionary or class string alias
        :return: initialized subject class
        """
        builder = self._builders.get(key)
        if not builder:
            logger.error(f'Model type {key} is not supported.')
            raise ValueError(f'Model type {key} is not supported.')
        return builder(**kwargs)
