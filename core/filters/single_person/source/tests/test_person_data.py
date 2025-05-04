import unittest
import numpy as np

from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.utils.cv.frames_segments import FramesSegments


class TestPersonData(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500
        self.maximum_sequence_length = 500

    def test_calculate_segments(self):
        for index in range(self.number_checks):
            number_segments = np.random.randint(1, 100)
            stride = np.random.randint(1, 10)
            frames_segments_ground_truth = self.frames_segments(number_segments, stride=stride)
            tracking_data = self.generate_tracking_data_with_known_segments(frames_segments_ground_truth, stride)
            frames_segments = tracking_data.calculate_segments(stride)
            self.assertTrue(np.all(frames_segments_ground_truth.values == frames_segments.values))


    @staticmethod
    def frames_segments(number_segments, stride=1, high_value = 1_000) -> FramesSegments:
        random_integers = np.random.randint(stride + 1, high=high_value, size=(number_segments * 2,))
        time_points = np.cumsum(random_integers)
        time_points = (time_points // stride) * stride
        segments = time_points.reshape((-1, 2))
        segments[:, 1] += 1
        return FramesSegments(segments)


    @staticmethod
    def generate_tracking_data_with_known_segments(segments: FramesSegments, stride=1) -> PersonTrackedData:
        tracking_data = PersonTrackedData()
        for segment in segments:
            current_number_indices = int((segment[1] - segment[0] - 1 ) / stride)
            current_segment_indices = np.linspace(segment[0], segment[1] - 1, current_number_indices + 1, endpoint=True)
            current_segment_indices = current_segment_indices.astype(np.int64)
            current_segment_indices = current_segment_indices[:-1]
            current_number_boxes = len(current_segment_indices)
            current_bounding_boxes = TestPersonData.bounding_boxes(current_number_boxes).astype(np.int64)

            current_confidences = 0.5 * np.ones(current_segment_indices.shape)
            tracking_data._bounding_boxes.extend(current_bounding_boxes)
            tracking_data._frames_indices._values = np.append(tracking_data._frames_indices._values, current_segment_indices).astype(np.int64)
            tracking_data._confidences = np.append(tracking_data._confidences, current_confidences)

        return tracking_data


    @staticmethod
    def bounding_boxes(number_boxes: int):
        boxes_array_top_left = np.random.randint(-2048, 2048, (number_boxes, 2))
        boxes_array_width_height = np.random.randint(1, 500, (number_boxes, 2))
        boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))
        return boxes_array.astype(np.int64)