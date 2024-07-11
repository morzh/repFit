from loguru import logger


class PersonsMaskFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            logger.error(f'Model type {key} is not supported for persons mask.')
            raise ValueError(f'Model type {key} is not supported for persons mask.')
        return builder(**kwargs)
