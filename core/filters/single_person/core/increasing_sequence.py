from multiprocessing.managers import Value

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
            ValueError('Only integer indices supported.')
        else:
            indices.sort()
            self._values = indices

    def append(self, index: int):
        """

        """
        if self._values.size == 0:
            self._values = np.array([index])
        elif self._values.shape[0] > 0 and index > self._values[-1]:
            self._values = np.append(self._values, index)
        else:
            raise ValueError('New frame index should be greater, than the previous one.')


    def __getitem__(self, item):
        return self._values[item]


    def __setitem__(self, key, value):
        raise PermissionError('Values are not writable')

    def is_consistent(self):
        derivatives = self._values[1:] - self._values[:-1]
        if np.alltrue(derivatives > 0):
            return True
        return False

    @property
    def values(self):
        return self._values



