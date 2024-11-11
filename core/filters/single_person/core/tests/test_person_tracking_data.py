import unittest
import numpy as np

from core.filters.single_person.core.strictly_increasing_sequence import StrictlyIncreasingSequence
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray
from core.filters.single_person.core.person_tracking_data import PersonTrackingData


class TestPersonTrackingData(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500

    def test_append_correct(self):
        tracking_data = self.fill_data()
        for index in range(self.number_checks):
            top_lefts = np.random.randint(-500, 500, (1, 2))
            width_heights = np.random.randint(1, 500, (1, 2))
            bounding_box = np.hstack((top_lefts, width_heights)).reshape(4,)
            frame_index = tracking_data._frames_indices._values[-1] + np.random.randint(1, 20)
            confidence = np.random.random((1,))

            tracking_data.append(bounding_box, frame_index, confidence)


    def test_append_incorrect_bounding_box(self):
        tracking_data = self.fill_data()
        for index in range(self.number_checks):
            top_lefts = np.random.randint(-500, 500, (1, 2))
            width_heights = np.random.randint(-200, 220, (1, 2))

            if np.alltrue(width_heights > 0):
                width_heights[0, 0] *= -1

            bounding_box = np.hstack((top_lefts, width_heights)).reshape(4,)
            frame_index =  np.random.randint(1, 20000)
            confidence = np.random.random((1,))

            with self.assertRaises(ValueError):
                tracking_data.append(bounding_box, frame_index, confidence)


    def test_append_incorrect_frame_index(self):
        tracking_data = self.fill_data()
        for index in range(self.number_checks):
            top_lefts = np.random.randint(-500, 500, (1, 2))
            width_heights = np.random.randint(1, 220, (1, 2))

            bounding_box = np.hstack((top_lefts, width_heights)).reshape(4,)
            frame_index =  np.random.randint(1, tracking_data._frames_indices._values[-1])
            confidence = np.random.random((1,))

            with self.assertRaises(ValueError):
                tracking_data.append(bounding_box, frame_index, confidence)


    def test_bounding_box(self):
        ...

    def test_calculate_segments(self):
        ...

    @staticmethod
    def fill_data() -> PersonTrackingData:
        number_occurrences = 5_000
        number_frames = 10_000

        top_lefts = np.random.randint(-500, 500, (number_occurrences, 2))
        width_heights = np.random.randint(1, 500, (number_occurrences, 2))

        bounding_boxes = np.hstack((top_lefts, width_heights))
        confidences = np.random.random((number_occurrences,))
        indices = np.random.randint(0, number_frames, (number_occurrences,))
        indices.sort()

        bounding_boxes_2d_array = BoundingBoxes2DArray(bounding_boxes)
        frames_indices = StrictlyIncreasingSequence(indices)

        tracking_data = PersonTrackingData()
        tracking_data._bounding_boxes = bounding_boxes_2d_array
        tracking_data._frames_indices = frames_indices
        tracking_data._confidences = confidences

        return tracking_data