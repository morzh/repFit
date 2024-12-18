import unittest
import numpy as np

from core.utils.geometry.segments.aligned_segment_2d import AlignedSegment2D, AlignedSegmentType


class TestAlignedSegment(unittest.TestCase):

    def setUp(self):
        self.number_checks = 4_500


    def test_is_less_greater_than(self):
        for _ in range(self.number_checks):
            current_segment = self.generate_segment(AlignedSegmentType.VERTICAL)
            current_segment_start = current_segment.y
            current_segment_end = current_segment.y + current_segment.length
            current_segment_intermediate = (1e-8 + 0.999999 * np.random.random()) * current_segment.length + current_segment_start

            self.assertFalse(current_segment.is_less_than(current_segment_start))
            self.assertFalse(current_segment.is_less_than(current_segment_intermediate))
            self.assertFalse(current_segment.is_less_than(current_segment_end))
            self.assertTrue(current_segment.is_less_than(current_segment_end + 1e-8))

            self.assertTrue(current_segment.is_greater_than(current_segment_start - 1e-8))
            self.assertFalse(current_segment.is_greater_than(current_segment_start))
            self.assertFalse(current_segment.is_greater_than(current_segment_intermediate))
            self.assertFalse(current_segment.is_greater_than(current_segment_end))


    @staticmethod
    def generate_segment(segment_type: AlignedSegmentType = AlignedSegmentType.VERTICAL):
        x = np.random.randint(-10_000, 10_000, 1)
        y = np.random.randint(-10_000, 10_000, 1)
        length = np.random.randint(1, 2500, 1)
        return AlignedSegment2D(x, y, length, segment_type)