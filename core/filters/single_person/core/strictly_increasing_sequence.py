import numpy as np


class StrictlyIncreasingSequence:
    """
    Description:
        Increasing integer sequence class.
    """
    __slots__ = ['_values']

    def __init__(self, indices: np.ndarray | None = None):
        if indices is None:
            self._values: np.ndarray = np.array([])
        elif not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Only integer indices supported.')
        elif not self.is_consistent(indices):
            raise ValueError('Indices sequence is not strictly increasing.')
        else:
            self._values = indices


    def __getitem__(self, item):
        return self._values[item]


    def __setitem__(self, key, value):
        raise PermissionError('Values are not writable')


    def __len__(self) -> int:
        return self._values.shape[0]


    def append(self, index: int):
        """
        Description:

        :param index:
        """
        if self._values.size == 0:
            self._values = np.array([index])
        elif self._values.shape[0] > 0 and index > self._values[-1]:
            self._values = np.append(self._values, index)
        else:
            raise ValueError('New frame index should be greater, than the previous one.')

    @staticmethod
    def is_consistent(indices):
        """
        Description:

        :param indices:
        """
        if indices.shape[0] == 1:
            return True

        derivatives = indices[1:] - indices[:-1]
        if np.alltrue(derivatives > 0):
            return True

        return False

    @property
    def values(self):
        """
        Description:

        """
        return self._values



