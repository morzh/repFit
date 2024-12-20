import numpy as np
import unittest

from rich.segment import Segments

from core.utils.cv.frames_segments import FramesSegments


class TestFramesSegments(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500


    def test_init_no_arguments(self):
        test_segments = FramesSegments(None)
        self.assertTrue(test_segments.shape == (0, 2))
        test_segments = FramesSegments()
        self.assertTrue(test_segments.shape == (0, 2))


    def test_init_consistent(self):
        number_checks = 1500
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 1_000)
            consistent_segments = self.generate_consistent_segments(number_segments)
            test_segments = FramesSegments(consistent_segments)
            self.assertTrue(test_segments.values.shape == consistent_segments.shape)


    def test_init_inconsistent(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 1_000)
            inconsistent_segments = self.generate_inconsistent_segments(number_segments)

            with self.assertWarns(UserWarning) as _:
                test_segments = FramesSegments(inconsistent_segments)

            self.assertTrue(test_segments.values.shape == (0, 2))


    def test_getitem(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 1_000)
            segments_array = self.generate_consistent_segments(number_segments, 1, number_segments)
            segments = FramesSegments(segments_array)

            segment_index = np.random.randint(1, number_segments)
            self.assertTrue(np.alltrue(segments_array[segment_index] == segments[segment_index]))
            self.assertEqual(segments_array[segment_index, 0], segments[segment_index, 0])
            self.assertEqual(segments_array[segment_index, 1], segments[segment_index, 1])


    def test_append_segment_consistent(self):
        segments = FramesSegments(np.array([[0, 1]]))
        for _ in range(self.number_checks):
            new_segment_length = np.random.randint(0, 600)
            new_segment_gap = np.random.randint(0, 600)
            new_segment_start = int(segments.values[-1, -1]) + new_segment_gap
            new_segment = np.array([new_segment_start, new_segment_start + new_segment_length])
            segments.append_segment(new_segment)


    def test_append_segment_inconsistent(self):
        segments = FramesSegments(np.array([[0, 1000]]))
        for _ in range(self.number_checks):
            new_segment_length = np.random.randint(0, 600)
            new_segment_start = np.random.randint(0, 600)
            if new_segment_start > segments[-1, 0]:
                new_segment_start = int(segments[-1, -1]) - 1
            new_segment = np.array([new_segment_start, new_segment_start + new_segment_length])
            with self.assertRaises(ValueError):
                segments.append_segment(new_segment)


    def test_filter_by_length_filter_all(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            length_threshold = np.random.randint(1, 600)

            segments = self.generate_consistent_segments_with_upper_length(length_threshold, number_segments)
            segments_ = FramesSegments(segments)
            segments_.filter_by_length(length_threshold)

            self.assertEqual(segments_.size, 0)


    def test_filter_by_length_filter_none(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            length_threshold = np.random.randint(1, 600)

            segments_values = self.generate_consistent_segments_with_lower_length(length_threshold + 1, number_segments)
            segments = FramesSegments(segments_values)
            segments.filter_by_length(length_threshold)

            self.assertEqual(segments.size, segments_values.size)


    def test_filter_by_length(self):
        for _ in range(self.number_checks):
            number_segments = 2 * np.random.randint(1, 1_500)
            length_threshold = np.random.randint(1, 600)

            segments_lower_threshold = self.generate_consistent_segments_with_lower_length(length_threshold + 1, int(number_segments / 2))
            segments_upper_threshold = self.generate_consistent_segments_with_upper_length(length_threshold, int(number_segments / 2))
            segments_upper_threshold += segments_lower_threshold[-1, -1]

            segments_values = np.vstack((segments_lower_threshold, segments_upper_threshold))
            segments = FramesSegments(segments_values)
            segments.filter_by_length(length_threshold)

            self.assertEqual(segments.size, segments_values.size / 2)


    def test_complements_correct_bounds(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            segments_array = self.generate_consistent_segments(number_segments)
            segments = FramesSegments(segments_array)
            high_bound = segments[-1, -1] + np.random.randint(0, 50)
            low_bound = segments[0, 0] - np.random.randint(0, 50)

            segments.complement(lower_bound=low_bound, upper_bound=high_bound)
            segments.complement(lower_bound=low_bound, upper_bound=high_bound)

            self.assertTrue(np.alltrue(segments_array == segments.values))


    def test_complements_equal_endpoints(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 10)
            segments_equal_endpoints = self.generate_segments_equal_endpoints(number_segments)
            segments = FramesSegments(segments_equal_endpoints)

            low_bound = int(segments_equal_endpoints[0, 0])
            high_bound = int(segments_equal_endpoints[-1, -1])

            segments.complement(lower_bound=low_bound, upper_bound=high_bound)
            segments.complement(lower_bound=low_bound, upper_bound=high_bound)

            self.assertTrue(np.alltrue(segments_equal_endpoints == segments.values))


    def test_compliment_incorrect_bounds(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            segments_array = self.generate_consistent_segments(number_segments)
            segments = FramesSegments(segments_array)
            high_bound = segments[-1, -1] - np.random.randint(1, 50)
            low_bound = segments[0, 0] + np.random.randint(1, 50)

            with self.assertRaises(ValueError):
                segments.complement(lower_bound=low_bound, upper_bound=high_bound)


    def test_filter_degenerate(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 20)
            segments_array = self.generate_consistent_segments(number_segments)
            segments_degenerate = self.add_degenerate_segments(segments_array)

            segments = FramesSegments(segments_degenerate)
            segments.filter_degenerate()
            segments.combine_adjacent()

            self.assertTrue(np.alltrue(segments_array == segments.values))


    def test_bridge_gaps(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 200)
            segments_array = self.generate_consistent_segments(number_segments, low_value=3)
            gaps_lengths = np.abs(segments_array[1:, 0] - segments_array[:-1, 1])
            maximum_gap = np.min(gaps_lengths) - 1
            if maximum_gap == 0: continue
            segments_array_gaps = self.add_gaps(segments_array, maximum_gap)

            segments = FramesSegments(segments_array_gaps)
            segments.bridge_gaps(maximum_gap)

            self.assertTrue(np.alltrue(segments_array == segments.values))


    def test_combine_adjacent(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(2, 1_000)
            segments_array = self.generate_consistent_segments(number_segments, 2, 600)
            segments_array_adjacent = self.insert_adjacent_segments(segments_array)

            segments = FramesSegments(segments_array_adjacent)
            segments.combine_adjacent()

            self.assertTrue(np.alltrue(segments_array == segments.values))


    def test_check_consistency(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)

            consistent_segments = self.generate_consistent_segments(number_segments, low_value=1, high_value=12_000)
            inconsistent_segments = self.generate_inconsistent_segments(number_segments, low_value=1, high_value=12_000)

            self.assertTrue(FramesSegments._check_consistency(consistent_segments))
            self.assertFalse(FramesSegments._check_consistency(inconsistent_segments))


    def test_lengths(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            apriori_lengths = np.random.randint(0, high=3000, size=(number_segments,))

            segments_values = self.generate_consistent_segments_with_given_lengths(apriori_lengths, number_segments)
            segments = FramesSegments(segments_values)

            self.assertTrue(np.alltrue(apriori_lengths == segments.lengths))


    def test_as_frame_indices(self):
        for _ in range(self.number_checks):
            number_segments = np.random.randint(1, 1_500)
            current_segments = FramesSegments(self.generate_consistent_segments(number_segments, low_value=1, high_value=10))
            current_segments_frames_indices = current_segments.frames_indices()

            for segments_index, segment_endpoints in enumerate(current_segments_frames_indices):
                current_check_difference = segment_endpoints[1:] - segment_endpoints[:-1]
                self.assertTrue(np.alltrue(current_check_difference == 1))
                self.assertEqual(current_segments_frames_indices[segments_index][0], segment_endpoints[0])
                self.assertEqual(current_segments_frames_indices[segments_index][-1], current_segments[segments_index, 1] - 1)


    @staticmethod
    def generate_consistent_segments_with_upper_length(length_threshold, number_segments) -> np.ndarray:
        apriori_lengths = np.random.randint(0, high=length_threshold, size=(number_segments,))
        apriori_gaps =  np.random.randint(0, high=600, size=(number_segments,))

        pre_segments = np.empty((number_segments, 2))
        pre_segments[:, 0] = apriori_gaps
        pre_segments[:, 1] = apriori_lengths
        pre_segments = pre_segments.flatten()

        segments = np.cumsum(pre_segments).reshape((-1, 2))
        return segments


    @staticmethod
    def generate_consistent_segments_with_lower_length(length_threshold, number_segments) -> np.ndarray:
        apriori_lengths = np.random.randint(length_threshold, high=length_threshold + 600, size=(number_segments,))
        apriori_gaps =  np.random.randint(0, high=600, size=(number_segments,))

        pre_segments = np.empty((number_segments, 2))
        pre_segments[:, 0] = apriori_gaps
        pre_segments[:, 1] = apriori_lengths
        pre_segments = pre_segments.flatten()

        segments = np.cumsum(pre_segments).reshape((-1, 2))
        return segments


    @staticmethod
    def generate_consistent_segments_with_given_lengths(lengths, number_segments, low_value=0, high_value=600) -> np.ndarray:
        apriori_gaps = np.random.randint(low_value, high=high_value, size=(number_segments,))

        pre_segments = np.empty((number_segments, 2), dtype=np.int64)
        pre_segments[:, 0] = apriori_gaps
        pre_segments[:, 1] = lengths
        pre_segments = pre_segments.flatten()

        segments = np.cumsum(pre_segments).reshape((-1, 2))
        return segments


    @staticmethod
    def generate_consistent_segments(number_segments, low_value = 1, high_value = 1_000) -> np.ndarray:
        random_integers = np.random.randint(low_value, high=high_value, size=(number_segments * 2,))
        time_points = np.cumsum(random_integers)
        consistent_segments = time_points.reshape((-1, 2))
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


    @staticmethod
    def insert_adjacent_segments(segments) -> np.ndarray:
        adjacent_segments = np.empty((0, 2), dtype=np.int64)
        for index in range(segments.shape[0]):
            insert_choice = np.random.choice(a=[False, True], size=(1,))[0]
            if insert_choice and segments[index, 1] - segments[index, 0] >= 3:
                center_value = int(0.5 * (segments[index, 1] + segments[index, 0]))
                first_segment = np.array([segments[index, 0], center_value])
                second_segment = np.array([center_value, segments[index, 1]])
                adjacent_segments = np.vstack((adjacent_segments, first_segment))
                adjacent_segments = np.vstack((adjacent_segments, second_segment))
            else:
                adjacent_segments = np.vstack((adjacent_segments, segments[index]))

        return  adjacent_segments


    @staticmethod
    def generate_segments_equal_endpoints(number_segments, low_value = 1, high_value = 1_000):
        random_integers = np.random.randint(low_value, high=high_value, size=(number_segments + 1,))
        time_points = np.cumsum(random_integers)
        segments = np.empty((number_segments, 2), dtype=np.int32)
        segments[:, 0] = time_points[:-1]
        segments[:, 1] = time_points[1:]
        return segments


    @staticmethod
    def add_degenerate_segments(segments: np.ndarray) -> np.ndarray:
        segments_degenerate = np.empty((0, 2), dtype=np.int64)
        for index in range(segments.shape[0]):
            insert_choice = np.random.choice(a=[False, True], size=(1,))[0]
            if insert_choice:
                center_value = int(0.5 * (segments[index, 1] + segments[index, 0]))
                first_segment = np.array([segments[index, 0], center_value])
                degenerate_segments = np.array([center_value, center_value])
                second_segment = np.array([center_value , segments[index, 1]])
                segments_degenerate = np.vstack((segments_degenerate, first_segment))
                segments_degenerate = np.vstack((segments_degenerate, degenerate_segments))
                segments_degenerate = np.vstack((segments_degenerate, second_segment))
            else:
                segments_degenerate = np.vstack((segments_degenerate, segments[index]))

        return segments_degenerate


    @staticmethod
    def add_gaps(segments, maximum_gap):
        segments_with_gaps = np.empty((0, 2), dtype=np.int64)
        for index in range(segments.shape[0]):
            insert_choice = np.random.choice(a=[False, True], size=(1,))[0]
            current_segment_length = segments[index, 1] - segments[index, 0]
            current_gap = np.random.randint(1, maximum_gap)
            if insert_choice and current_segment_length > current_gap:
                discrepancy = current_segment_length - current_gap
                value_first = int(np.floor(0.5 * discrepancy))
                value_second = int(np.ceil(0.5 * discrepancy))
                first_segment = np.array([segments[index, 0], segments[index, 0] + value_first])
                second_segment = np.array([segments[index, 1] - value_second, segments[index, 1]])
                segments_with_gaps = np.vstack((segments_with_gaps, first_segment))
                segments_with_gaps = np.vstack((segments_with_gaps, second_segment))
            else:
                segments_with_gaps = np.vstack((segments_with_gaps, segments[index]))

        return segments_with_gaps
