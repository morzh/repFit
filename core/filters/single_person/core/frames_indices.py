import numpy as np
import warnings

from pyqtgraph.examples.MultiDataPlot import values


class FramesIndices:
    """
    Description:
        Class storage for strictly increasing sequence of numbers.

    :ivar _values: sequence values.
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


    def __copy__(self):
        return FramesIndices(np.copy(self._values))


    def append(self, element: int) -> None:
        """
        Description:
            Appends new element to an existing sequence.

        :param element: element to append.
        """
        if self._values.size == 0:
            self._values = np.array([element])
        elif self._values.size > 0 and element > self._values[-1]:
            self._values = np.append(self._values, element)
        else:
            raise ValueError(f'New frame index should be greater, than the previous one. Got {element} <= {self._values[-1]}')


    @staticmethod
    def is_consistent(indices: np.ndarray) -> bool:
        """
        Description:
            Checks if indices 1D array is strictly increasing.

        :param indices: indices values

        :return: True if ``indices`` is strictly increasing sequence, False otherwise.
        """
        if indices.size != indices.flatten().shape[0]:
            raise ValueError('Indices should be a 1D array.')

        if indices.shape[0] == 1:
            return True

        derivatives = indices[1:] - indices[:-1]
        if np.alltrue(derivatives > 0):
            return True

        return False


    @property
    def values(self) -> np.ndarray:
        """
        Description:
            Indices values getter

        :return: indices array.
        """
        return self._values



