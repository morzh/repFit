import numpy as np
import unittest

from core.utils.cv.segments import Segments

class TestSegments(unittest.TestCase):

    # def setUp(self):
    #     self.segments = Segments()


    def test_init(self):
        number_checks = 500

        for index in range(number_checks):
            segments_candidates = np.random.randint((100, 2))
            with self.assertWarns(Warning):
                segments = Segments(segments_candidates)

        for index in range(number_checks):
            increasing_numbers_sequence = [n for n in range(3000)]
            ...


        test_segments = Segments(None)
        self.assertEqual(test_segments.shape, (0, 2))
        test_segments = Segments()
        self.assertEqual(test_segments.shape, (0, 2))


    def test_getitem(self):
        ...

    def test_append_segment(self):
        segments = Segments()

    def test_filter_by_length(self):
        """

        """

    def test_complements(self):
        """

        """

    def test_bridge_gaps(self):
        ...

    def test_combine_adjacent(self):
        """

        """

    def test_check_consistency(self):
        number_checks = 3500
        for _ in range(number_checks):
            number_segments = np.random.randint(1, 1_500)

            consistent_segments = self.generate_consistent_segments(number_segments, low_value=1, high_value=12_000)
            self.assertTrue(Segments._check_consistency(consistent_segments))

            inconsistent_segments = self.generate_inconsistent_segments(number_segments, low_value=1, high_value=12_000)
            self.assertFalse(Segments._check_consistency(inconsistent_segments))


    def test_lengths(self):
        """

        """

    @staticmethod
    def generate_consistent_segments(number_segments, low_value = 1, high_value = 12_000) -> np.ndarray:
        random_integers = np.random.randint(low_value, high=high_value, size=(number_segments * 2,))
        random_integers.sort()
        consistent_segments = random_integers.reshape((-1, 2))
        return consistent_segments


    @staticmethod
    def generate_inconsistent_segments(number_segments, low_value = 1, high_value = 12_000) -> np.ndarray:
        random_integers = np.random.randint(low_value, high=high_value, size=(number_segments * 2,))
        random_integers_derivative = random_integers[1:] - random_integers[:-1]
        # sometimes random numbers generator produces sorted sequence of numbers. If it is so, just permutate first and last elements.
        if np.all(random_integers_derivative > 0):
            random_integers[0], random_integers[-1] = random_integers[-1], random_integers[0]

        inconsistent_segments = random_integers.reshape((-1, 2))
        return inconsistent_segments


