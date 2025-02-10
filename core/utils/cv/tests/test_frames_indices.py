import copy
import unittest
import numpy as np

from core.utils.cv.frames_indices import FramesIndices


class TestStrictlyIncreasingSequence(unittest.TestCase):

    def setUp(self):
        self.number_checks = 2_500
        self.maximum_sequence_length = 1500


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


    def test_add_dunder_methods(self):
        for _ in range(self.number_checks):
            current_apriori_known_frames_indices_values = self.correct_sequence(1, self.maximum_sequence_length)
            current_apriori_frames_indices_values_number = current_apriori_known_frames_indices_values.shape[0]
            current_frames_indices_values_1_mask = np.random.randint(0, 2, current_apriori_known_frames_indices_values.shape[0]).astype(bool)
            current_frames_indices_values_2_mask = ~current_frames_indices_values_1_mask.copy()
            #  Adding more frame indices values to two separate frames indices arrays. The idea is that two frame indices arrays should have non-empty overlap.
            current_frames_indices_values_1_mask = np.logical_or(current_frames_indices_values_1_mask, np.random.randint(0, 2, current_apriori_frames_indices_values_number).astype(bool))
            current_frames_indices_values_2_mask = np.logical_or(current_frames_indices_values_2_mask, np.random.randint(0, 2, current_apriori_frames_indices_values_number).astype(bool))

            current_frames_indices_apriori_sum = FramesIndices(current_apriori_known_frames_indices_values)
            current_frame_indices_1 = FramesIndices(current_apriori_known_frames_indices_values[current_frames_indices_values_1_mask])
            current_frame_indices_2 = FramesIndices(current_apriori_known_frames_indices_values[current_frames_indices_values_2_mask])

            current_frame_indices_sum = current_frame_indices_1 + current_frame_indices_2
            current_frame_indices_1 += current_frame_indices_2

            self.assertTrue(current_frames_indices_apriori_sum == current_frame_indices_sum)
            self.assertTrue(current_frames_indices_apriori_sum == current_frame_indices_1)


    def test_sub_dunder_methods(self):
        for _ in range(self.number_checks):
            current_apriori_known_frames_indices_values = self.correct_sequence(1, self.maximum_sequence_length)
            current_frames_indices_values_1_mask = np.random.randint(0, 2, current_apriori_known_frames_indices_values.shape[0]).astype(bool)
            current_frames_indices_values_2_mask = ~current_frames_indices_values_1_mask.copy()

            current_frames_indices_apriori = FramesIndices(current_apriori_known_frames_indices_values)
            current_frame_indices_1 = FramesIndices(current_apriori_known_frames_indices_values[current_frames_indices_values_1_mask])
            current_frame_indices_2 = FramesIndices(current_apriori_known_frames_indices_values[current_frames_indices_values_2_mask])

            current_frame_indices_subtract_1 = current_frames_indices_apriori - current_frame_indices_1
            current_frame_indices_subtract_2 = current_frames_indices_apriori - current_frame_indices_2

            self.assertTrue(current_frame_indices_subtract_1 == current_frame_indices_2)
            self.assertTrue(current_frame_indices_subtract_2 == current_frame_indices_1)

            current_frames_indices_apriori -= current_frame_indices_1
            self.assertTrue(current_frames_indices_apriori == current_frame_indices_2)


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
