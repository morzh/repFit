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
            self._values: np.ndarray = np.array([], dtype=np.int64)
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


    def __eq__(self, other):
        return np.array_equal(self._values, other.values)


    def __add__(self, other):
        stacked_values = np.concatenate((self._values, other.values), dtype=np.int64)
        return FramesIndices(np.unique(stacked_values))


    def __iadd__(self, other):
        self._values = np.unique(np.concatenate((self._values, other.values)))
        return self

    def __iter__(self):
        return self._values.__iter__()


    def __sub__(self, other):
        difference_values = np.empty([])
        if isinstance(other, FramesIndices):
            difference_values = np.setdiff1d(self._values, other.values)
        elif isinstance(other, np.ndarray):
            difference_values = np.setdiff1d(self._values, other)

        return FramesIndices(difference_values)


    def __isub__(self, other):
        if isinstance(other, FramesIndices):
            self._values = np.setdiff1d(self._values, other.values)
            return self
        elif isinstance(other, np.ndarray):
            self._values = np.setdiff1d(self._values, other)
            return self


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


    def insert(self, new_frames_indices: np.ndarray) -> None:
        """
        Description:
            Insert new ``frames_indices`` to existing ones.

        :param new_frames_indices: frames indices
        """
        new_frames_indices = np.setdiff1d(new_frames_indices.flatten(), self._values)
        self._values = np.append(self._values, new_frames_indices)
        self._values = np.sort(self._values)


    def is_in_vicinity(self, input_frame: int, stride: int, mode='both') -> bool:
        """
        Description:

        @param input_frame: input frame index
        @param stride: video frames stride
        @param mode: mode of matching. Options are 'left', 'right' and 'both'.

        :return: True if input frame index is in vicinity of the sequence of frames. False otherwise.
        """
        closest_frame_index = (np.abs(self._values - input_frame)).argmin()
        closest_frame = self._values[closest_frame_index]
        match mode:
            case 'left':
                if 0 <= (input_frame - closest_frame) < stride: return True
            case 'right':
                if 0 <= (closest_frame - input_frame) < stride: return True
            case 'both':
                if abs(closest_frame - input_frame) < stride: return True
        return False

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



