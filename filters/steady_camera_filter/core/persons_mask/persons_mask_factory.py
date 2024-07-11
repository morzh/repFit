import typing

from loguru import logger


class PersonsMaskFactory:
    def __init__(self):
        """
        Description:
            Initializes empty builders dictionary
        """
        self._builders = {}

    def register_builder(self, key: str, builder: typing.Callable) -> None:
        """
        Description:
            Register persons mask builder. Builder creates class initialized with parameters.

        :param key: persons mask class key
        :param builder: persons mask class builder
        """
        self._builders[key] = builder

    def create(self, key: str, **kwargs) -> typing.Callable:
        """
        Description:
            This method return initialized by **kwargs class, defined bt key.
        :param key: class string alias
        """
        builder = self._builders.get(key)
        if not builder:
            logger.error(f'Model type {key} is not supported for persons mask.')
            raise ValueError(f'Model type {key} is not supported for persons mask.')
        return builder(**kwargs)
