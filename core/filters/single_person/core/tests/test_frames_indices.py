import copy
import unittest
import numpy as np

from core.filters.single_person.core.frames_indices import FramesIndices


class TestStrictlyIncreasingSequence(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500
        self.maximum_sequence_length = 500


    def test_init(self):
        # correct input sequence case
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(1, self.maximum_sequence_length)
            _ = FramesIndices(correct_sequence)
        # incorrect input sequence case
        for _ in range(self.number_checks):
            incorrect_sequence = self.incorrect_sequence(4, self.maximum_sequence_length)
            with self.assertRaises(ValueError):
                _ = FramesIndices(incorrect_sequence)


    def test_append_correct(self):
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(1, self.maximum_sequence_length)
            sequence = FramesIndices(correct_sequence)
            last_element = int(sequence.values[-1])
            addon = np.random.randint(1, 50)
            sequence.append(last_element + addon)


    def test_append_incorrect(self):
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(1, self.maximum_sequence_length)
            sequence = FramesIndices(correct_sequence)
            last_element = int(sequence.values[-1])
            addon = np.random.randint(-10, 0)
            with self.assertRaises(ValueError):
                sequence.append(last_element + addon)


    def test_getitem(self):
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(2, self.maximum_sequence_length)
            sequence = FramesIndices(correct_sequence)
            index = np.random.randint(0, len(sequence) - 1)
            _ = sequence[index]


    def test_setitem(self):
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(2, self.maximum_sequence_length)
            sequence = FramesIndices(correct_sequence)
            index = np.random.randint(0, len(sequence) - 1)
            with self.assertRaises(PermissionError):
                sequence[index] = 10

    def test_is_consistent(self):
        for _ in range(self.number_checks):
            correct_sequence = self.correct_sequence(4, self.maximum_sequence_length)
            incorrect_sequence = self.incorrect_sequence(4, self.maximum_sequence_length)

            is_consistent_false = FramesIndices.is_consistent(incorrect_sequence)
            is_consistent_true = FramesIndices.is_consistent(correct_sequence)

            self.assertFalse(is_consistent_false)
            self.assertTrue(is_consistent_true)


    @staticmethod
    def incorrect_sequence(low, high) -> np.ndarray:
        sequence_length = np.random.randint(low, high)
        sequence = np.random.randint(1, 50, (sequence_length,))
        correct_sequence = np.cumsum(sequence)
        incorrect_sequence = copy.deepcopy(correct_sequence)
        np.random.shuffle(incorrect_sequence)
        # sometimes np.random.shuffle leaves input array unchanged. Following check prevents this.
        if np.alltrue(correct_sequence == incorrect_sequence):
            incorrect_sequence[0] = correct_sequence[-1]
        return incorrect_sequence


    @staticmethod
    def correct_sequence(low, high) -> np.ndarray:
        sequence_length = np.random.randint(low, high)
        sequence = np.random.randint(1, 50, (sequence_length,))
        correct_sequence = np.cumsum(sequence)
        return correct_sequence
