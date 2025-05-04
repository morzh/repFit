import copy
import unittest
import numpy as np

from core.filters.single_person.source.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class TestConfidenceFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 3_500
        self.video_width = 1920
        self.video_height = 1080
        self.minimum_frames_number = 50
        self.minimum_box_area = 300
        self.maximum_box_area = int(0.5 * self.video_width * self.video_height)
        self.maximum_number_of_video_frames = 5000
        self.maximum_number_persons_in_tracks = 20
        self.maximum_frames_stride_value = 5
        self.maximum_area_threshold = 1200
        self.bounding_box_minimal_width = 30
        self.bounding_box_minimal_height = 30
        self.joints_number = 17
        self.confidence_range = 0.3, 0.7


    def test_confidence_filter_addon(self):
        for _ in range(self.number_checks):
            current_confidence_threshold_value = self.confidence_range[0] + (self.confidence_range[1] - self.confidence_range[0]) * np.random.rand()
            current_tracks_apriori = self.generate_tracks(confidence_range=(current_confidence_threshold_value, 1.0))
            current_tracks_to_filter = self.add_data_to_tracks(current_tracks_apriori, confidence_range=(0.05, current_confidence_threshold_value - 1e-6))

            current_full_body_filter = ConfidenceFilterAddon(confidence_threshold=current_confidence_threshold_value, filter_full_body_person=True)
            current_full_body_filter.process(current_tracks_to_filter)
            current_body_filter = ConfidenceFilterAddon(confidence_threshold=current_confidence_threshold_value, filter_full_body_person=False)
            current_body_filter.process(current_tracks_to_filter)

            for person_apriori, person_filtered in zip(current_tracks_apriori.persons.values(), current_tracks_to_filter.persons.values()):
                self.assertTrue(person_apriori.partial_body_data.frames_indices == person_filtered.partial_body_data.frames_indices)
                self.assertTrue(person_apriori.full_body_data.frames_indices == person_filtered.full_body_data.frames_indices)


    def generate_tracks(self, confidence_range=(0.25, 0.75)) -> MultiplePersonsTracks:
        persons_number = np.random.randint(1, self.maximum_number_persons_in_tracks)
        exact_video_frames_number = np.random.randint(self.minimum_frames_number, self.maximum_number_of_video_frames)
        video_frames_stride = np.random.randint(1, self.maximum_frames_stride_value)
        stride_frames_number = int(exact_video_frames_number / video_frames_stride)
        frames_indices = np.linspace(0, (exact_video_frames_number // video_frames_stride) * video_frames_stride, stride_frames_number + 1).astype(int)
        video_frames_number_inaccuracy = np.random.randint(-10, 10)
        approximate_video_frames_number = exact_video_frames_number + video_frames_number_inaccuracy
        fps = 20 + 40*np.random.random()

        video_properties = VideoProperties('path_to_video', self.video_width, self.video_height, approximate_video_frames_number, fps)
        multiple_persons_tracks = MultiplePersonsTracks(video_properties, stride=video_frames_stride, exact_frames_number=exact_video_frames_number)

        for person_index in range(persons_number):
            current_number_frames_indices = np.random.randint(int(frames_indices.shape[0] / 4), int(frames_indices.shape[0] / 2))
            current_frames_indices_mask = np.array(current_number_frames_indices * [True] + (frames_indices.shape[0] - current_number_frames_indices) * [False])
            np.random.shuffle(current_frames_indices_mask)
            current_frames_indices = frames_indices[current_frames_indices_mask]

            current_person_tracked_data = self.generate_person_tracked_data(current_frames_indices, confidence_range, multiple_persons_tracks)
            current_person = SinglePersonTrack()
            current_person.tracked_data = current_person_tracked_data
            current_person.is_active = True
            multiple_persons_tracks.persons[person_index] = current_person

            current_number_full_body_data_indices =  np.random.randint(0, current_frames_indices.shape[0])
            current_full_body_data_indices_mask = [True] * current_number_full_body_data_indices + [False] * (current_frames_indices.shape[0] - current_number_full_body_data_indices)
            np.random.shuffle(current_full_body_data_indices_mask)
            current_person.full_body_data.frames_indices = FramesIndices(current_frames_indices[current_full_body_data_indices_mask])

            current_number_data_indices =  np.random.randint(0, current_frames_indices.shape[0])
            current_data_indices_mask = [True] * current_number_data_indices + [False] * (current_frames_indices.shape[0] - current_number_data_indices)
            np.random.shuffle(current_data_indices_mask)
            current_person.partial_body_data.frames_indices = FramesIndices(current_frames_indices[current_data_indices_mask])

        return multiple_persons_tracks


    def generate_person_tracked_data(self, frames_indices: np.ndarray, confidence_range: tuple, tracks: MultiplePersonsTracks) -> PersonTrackedData:
        tracked_data = PersonTrackedData()

        for box_index, frame_index in enumerate(frames_indices):
            current_bounding_box_left = np.random.randint(0, int(tracks.video_properties.width / 2))
            current_bounding_box_top = np.random.randint(0, int(tracks.video_properties.height / 2))
            current_bounding_box_width = np.random.randint(self.bounding_box_minimal_width, self.video_width - current_bounding_box_left)
            current_bounding_box_height = np.random.randint(self.bounding_box_minimal_height, self.video_height - current_bounding_box_top)
            current_bounding_box = np.array([current_bounding_box_left, current_bounding_box_top, current_bounding_box_width, current_bounding_box_height])

            current_confidence = confidence_range[0] +  (confidence_range[1] - confidence_range[0])*np.random.random()

            tracked_data.append(current_bounding_box, int(frame_index), current_confidence)

        return tracked_data


    def add_data_to_tracks(self, tracks: MultiplePersonsTracks, confidence_range=(0.05, 0.45)) -> MultiplePersonsTracks:
        stride_frames_number = int(tracks.exact_frames_number / tracks.frames_stride)
        frames_indices = np.linspace(0, (tracks.exact_frames_number // tracks.frames_stride) * tracks.frames_stride, stride_frames_number + 1).astype(int)

        new_tracks = copy.deepcopy(tracks)

        for person in new_tracks.persons.values():
            current_person_candidate_frame_indices = np.setdiff1d(frames_indices, person.tracked_data.frames_indices)
            current_number_new_tracked_frames_indices = current_person_candidate_frame_indices.shape[0]
            if current_number_new_tracked_frames_indices:
                current_new_number_frames_indices = np.random.randint(0, current_number_new_tracked_frames_indices)
                if current_new_number_frames_indices:
                    current_frames_indices_mask = np.array(current_new_number_frames_indices * [True] + (current_person_candidate_frame_indices.shape[0] - current_new_number_frames_indices) * [False])
                    np.random.shuffle(current_frames_indices_mask)
                    current_new_frames_indices = current_person_candidate_frame_indices[current_frames_indices_mask]
                    current_new_number_of_data_elements = current_new_frames_indices.shape[0]

                    current_new_bounding_boxes = self.generate_bounding_boxes(current_new_number_of_data_elements, tracks.video_properties)
                    current_new_confidences = confidence_range[0] + (confidence_range[1] - confidence_range[0]) * np.random.random((current_new_number_of_data_elements,))
                    person.tracked_data.insert(current_new_frames_indices, current_new_bounding_boxes, current_new_confidences)

                    current_new_number_data_frames_indices = np.random.randint(0, np.maximum(current_new_number_of_data_elements, 1))
                    current_data_frames_indices_mask = np.array(current_new_number_data_frames_indices * [True] + (current_new_number_of_data_elements - current_new_number_data_frames_indices) * [False])
                    np.random.shuffle(current_data_frames_indices_mask)
                    current_new_data_indices = current_new_frames_indices[current_data_frames_indices_mask]
                    person.partial_body_data.insert(current_new_data_indices)

                    current_number_full_body_data_frames_indices = np.random.randint(0, np.maximum(current_new_number_of_data_elements, 1))
                    current_full_body_data_frames_indices_mask = np.array(current_number_full_body_data_frames_indices * [True] + (current_new_number_of_data_elements - current_number_full_body_data_frames_indices) * [False])
                    np.random.shuffle(current_full_body_data_frames_indices_mask)
                    current_new_full_body_data_indices = current_new_frames_indices[current_full_body_data_frames_indices_mask]
                    person.full_body_data.insert(current_new_full_body_data_indices)

        return new_tracks


    @staticmethod
    def generate_bounding_boxes(number_boxes: int, video_properties: VideoProperties) -> BoundingBoxes2DArray:
        if not number_boxes:
            return BoundingBoxes2DArray()

        lefts = np.random.randint(0, video_properties.width, (number_boxes, 1))
        tops = np.random.randint(0, video_properties.height, (number_boxes, 1))
        widths = np.random.randint(1, int(0.5 * video_properties.width), (number_boxes, 1))
        heights = np.random.randint(1, int(0.5 * video_properties.height), (number_boxes, 1))

        boxes = np.hstack((lefts, tops, widths, heights))
        bounding_boxes = BoundingBoxes2DArray(boxes)
        return bounding_boxes



