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
            random_integers = np.random.random_integers(1, 12_000, (number_segments * 2,))
            random_integers.sort()
            consistent_segments = random_integers.reshape((-1, 2))
            segments_are_consistent = Segments._check_consistency(consistent_segments)
            self.assertTrue(segments_are_consistent)

        for _ in range(number_checks):
            number_segments = np.random.randint(1, 1_000)
            random_integers = np.random.random_integers(1, 10_000, (number_segments * 2,))
            # sometimes random numbers generator produces sorted sequence of numbers. If it is so, just permutate first and last elements.
            random_integers_derivative = random_integers[1:] - random_integers[:-1]
            if np.all(random_integers_derivative > 0):
                random_integers[0], random_integers[-1] = random_integers[-1], random_integers[0]

            inconsistent_segments = random_integers.reshape((-1, 2))
            segments_are_consistent = Segments._check_consistency(inconsistent_segments)
            self.assertFalse(segments_are_consistent)


    def test_lengths(self):
        """

        """
