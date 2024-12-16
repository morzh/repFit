import unittest
import numpy as np

from core.filters.single_person.core.strictly_increasing_sequence import StrictlyIncreasingSequence
from core.utils.cv.frames_segments import FramesSegments
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray
from core.filters.single_person.core.person_tracking_data import PersonTrackingData


class TestPersonTrackingData(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500

    def test_append_correct(self):
        tracking_data = self.generate_tracking_data()
        for index in range(self.number_checks):
            top_lefts = np.random.randint(-500, 500, (1, 2))
            width_heights = np.random.randint(1, 500, (1, 2))
            bounding_box = np.hstack((top_lefts, width_heights)).reshape(4,)
            frame_index = tracking_data._frames_indices._values[-1] + np.random.randint(1, 20)
            confidence = np.random.random((1,))

            tracking_data.append(bounding_box, frame_index, confidence)


    def test_append_incorrect_bounding_box(self):
        tracking_data = self.generate_tracking_data()
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
        tracking_data = self.generate_tracking_data()
        for index in range(self.number_checks):
            top_lefts = np.random.randint(-500, 500, (1, 2))
            width_heights = np.random.randint(1, 220, (1, 2))

            bounding_box = np.hstack((top_lefts, width_heights)).reshape(4,)
            frame_index =  np.random.randint(1, tracking_data._frames_indices._values[-1])
            confidence = np.random.random((1,))

            with self.assertRaises(ValueError):
                tracking_data.append(bounding_box, frame_index, confidence)


    def test_bounding_box(self):
        for _ in range(self.number_checks):
            x_min =  np.random.randint(-500, 500)
            y_min =  np.random.randint(-500, 500)

            x_max =  x_min + np.random.randint(0, 500)
            y_max =  y_min + np.random.randint(0, 500)

            width_1 = np.random.randint(1, 500)
            width_2 = np.random.randint(1, 500)

            height_1 = np.random.randint(1, 500)
            height_2 = np.random.randint(1, 500)

            number_steps = np.random.randint(3, 20)

            frame_1 = np.random.randint(0, 500)
            frame_2 = frame_1 + number_steps

            xs = np.linspace(x_min, x_max, number_steps + 1).reshape((-1, 1))
            ys = np.linspace(y_min, y_max, number_steps + 1).reshape((-1, 1))
            widths = np.linspace(width_1, width_2, number_steps + 1).reshape((-1, 1))
            heights = np.linspace(height_1, height_2, number_steps + 1).reshape((-1, 1))

            bounding_boxes_interpolated = np.hstack((xs, ys, widths, heights))
            frames = np.linspace(frame_1, frame_2, number_steps + 1).astype(int)
            confidences = np.random.rand(number_steps)

            tracking_data = PersonTrackingData()
            tracking_data.append(bounding_boxes_interpolated[0], int(frames[0]), float(confidences[0]))
            tracking_data.append(bounding_boxes_interpolated[-1], int(frames[-1]), float(confidences[-1]))

            for index in range(frames.shape[0]):
                frame_index = frames[index]
                interpolated_bounding_box = tracking_data.bounding_box(frame_index)
                self.assertTrue(np.alltrue(np.isclose(interpolated_bounding_box, bounding_boxes_interpolated[index], atol=1e-18)))


    def test_calculate_segments(self):
        for index in range(self.number_checks):
            number_segments = np.random.randint(1, 100)
            stride = np.random.randint(1, 10)
            frames_segments_ground_truth = self.frames_segments(number_segments, stride=stride)
            tracking_data = self.generate_tracking_data_with_known_segments(frames_segments_ground_truth, stride)
            frames_segments = tracking_data.calculate_segments(stride)
            self.assertTrue(np.alltrue(frames_segments_ground_truth.values == frames_segments.values))


    @staticmethod
    def generate_tracking_data() -> PersonTrackingData:
        number_occurrences = 5_000

        top_lefts = np.random.randint(-500, 500, (number_occurrences, 2))
        width_heights = np.random.randint(1, 500, (number_occurrences, 2))

        bounding_boxes = np.hstack((top_lefts, width_heights))
        confidences = np.random.random((number_occurrences,))
        indices = np.random.randint(1, 15, (number_occurrences,))
        indices  = np.cumsum(indices)

        bounding_boxes_2d_array = BoundingBoxes2DArray(bounding_boxes)
        frames_indices = StrictlyIncreasingSequence(indices)

        tracking_data = PersonTrackingData()
        tracking_data._bounding_boxes = bounding_boxes_2d_array
        tracking_data._frames_indices = frames_indices
        tracking_data._confidences = confidences

        return tracking_data


    @staticmethod
    def frames_segments(number_segments, stride=1, high_value = 1_000) -> FramesSegments:
        random_integers = np.random.randint(stride + 1, high=high_value, size=(number_segments * 2,))
        time_points = np.cumsum(random_integers)
        time_points = (time_points // stride) * stride
        segments = time_points.reshape((-1, 2))
        segments[:, 1] += 1
        return FramesSegments(segments)


    @staticmethod
    def generate_tracking_data_with_known_segments(segments: FramesSegments, stride=1) -> PersonTrackingData:
        tracking_data = PersonTrackingData()
        for segment in segments:
            current_number_indices = int((segment[1] - segment[0] - 1 ) / stride)
            current_segment_indices = np.linspace(segment[0], segment[1] - 1, current_number_indices + 1, endpoint=True)
            current_segment_indices = current_segment_indices.astype(np.int64)
            current_segment_indices = current_segment_indices[:-1]
            current_number_boxes = len(current_segment_indices)
            current_bounding_boxes = TestPersonTrackingData.bounding_boxes(current_number_boxes).astype(np.int64)

            current_confidences = 0.5 * np.ones(current_segment_indices.shape)
            tracking_data._bounding_boxes.extend(current_bounding_boxes)
            tracking_data._frames_indices._values = np.append(tracking_data._frames_indices._values, current_segment_indices).astype(np.int64)
            tracking_data._confidences = np.append(tracking_data._confidences, current_confidences)
            # for bounding_box, frame_index in zip(current_bounding_boxes, current_segment_indices):
            #     tracking_data.append(bounding_box, frame_index, 0.5)

        return tracking_data


    @staticmethod
    def single_bounding_box() -> np.ndarray:
        return np.array([*np.random.randint(-2048, 2048), *np.random.randint(1, 1024, (2,))])


    @staticmethod
    def bounding_boxes(number_boxes: int):
        boxes_array_top_left = np.random.randint(-2048, 2048, (number_boxes, 2))
        boxes_array_width_height = np.random.randint(1, 500, (number_boxes, 2))
        boxes_array = np.hstack((boxes_array_top_left, boxes_array_width_height))
        return boxes_array.astype(np.int64)
