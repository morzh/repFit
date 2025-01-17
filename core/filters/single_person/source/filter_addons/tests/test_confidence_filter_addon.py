import unittest
import numpy as np

from core.filters.single_person.source.filter_addons.absolute_area_filter_addon import AbsoluteAreaFilterAddon
from core.filters.single_person.source.filter_addons.confidence_filter_addon import ConfidenceFilterAddon
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_properties import VideoProperties


class TestConfidenceFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500
        self.video_width = 1920
        self.video_height = 1080
        self.minimum_frames_number = 50
        self.minimum_box_area = 300
        self.maximum_box_area = int(0.5 * self.video_width * self.video_height)
        self.maximum_number_of_video_frames = 5000
        self.number_persons_in_tracks = 20
        self.maximum_frames_stride_value = 5
        self.maximum_area_threshold = 1200
        self.bounding_box_minimal_width = 30
        self.bounding_box_minimal_height = 30
        self.joints_number = 17


    def test_confidence_filter_full_body(self):
        for _ in range(self.number_checks):
            current_tracks, current_confidences = self.generate_tracks(full_body_confidences=True)
            current_confidence_threshold = np.percentile(current_confidences, 50.0)
            current_confidences_mask = current_confidences >= current_confidence_threshold

            current_filter = ConfidenceFilterAddon(confidence_threshold=current_confidence_threshold)
            current_filter.process(current_tracks, filter_full_body_person=True)

            for person_id, person_track in current_tracks.persons.items():
                self.assertTrue(bool(len(person_track.data.frames_indices)) == current_confidences_mask[person_id])


    def test_confidence_filter_non_full_body(self):
        for _ in range(self.number_checks):
            current_tracks, current_confidences = self.generate_tracks(full_body_confidences=False)
            current_confidence_threshold = np.percentile(current_confidences, 50.0)
            current_confidences_mask = current_confidences >= current_confidence_threshold

            current_filter = ConfidenceFilterAddon(confidence_threshold=current_confidence_threshold)
            current_filter.process(current_tracks, filter_full_body_person=False)

            for person_id, person_track in current_tracks.persons.items():
                self.assertTrue(bool(len(person_track.data.frames_indices)) == current_confidences_mask[person_id])


    def generate_tracks(self, full_body_confidences=False) -> tuple[MultiplePersonsTracks, list[np.ndarray]]:
        persons_number = np.random.randint(1, self.number_persons_in_tracks)
        exact_video_frames_number = np.random.randint(self.minimum_frames_number, self.maximum_number_of_video_frames)
        video_frames_stride = np.random.randint(1, self.maximum_frames_stride_value)
        stride_frames_number = int(exact_video_frames_number / video_frames_stride)
        frames_indices = np.linspace(0, (exact_video_frames_number // video_frames_stride) * video_frames_stride, stride_frames_number + 1).astype(int)
        video_frames_number_inaccuracy = np.random.randint(-10, 10)
        approximate_video_frames_number = exact_video_frames_number + video_frames_number_inaccuracy
        fps = 20 + 40*np.random.random()

        video_properties = VideoProperties('path_to_video', self.video_width, self.video_height, approximate_video_frames_number, fps)
        multiple_persons_tracks = MultiplePersonsTracks(video_properties, stride=video_frames_stride, exact_frames_number=exact_video_frames_number)
        tracks_confidences = []

        for person_index in range(persons_number):
            current_number_frames_indices = np.random.randint(0, frames_indices.shape[0])
            current_frames_indices_mask = np.array(current_number_frames_indices * [True] + (frames_indices.shape[0] - current_number_frames_indices) * [False])
            np.random.shuffle(current_frames_indices_mask)
            current_frames_indices = frames_indices[current_frames_indices_mask]

            current_person_tracked_data = self.generate_person_tracked_data(current_frames_indices, multiple_persons_tracks)
            current_person = SinglePersonTrack()
            current_person.tracked_data = current_person_tracked_data
            multiple_persons_tracks.persons[person_index] = current_person

            if full_body_confidences:
                current_number_full_body_data_indices =  np.random.randint(0, current_frames_indices.shape[0])
                current_full_body_data_indices_mask = [True] * current_number_full_body_data_indices + [False] * (current_frames_indices.shape[0] - current_number_full_body_data_indices)
                np.random.shuffle(current_full_body_data_indices_mask)
                current_person.full_body_data.frames_indices = current_frames_indices[current_full_body_data_indices_mask]
                tracks_confidences.append(current_frames_indices[current_full_body_data_indices_mask])
            else:
                current_number_data_indices =  np.random.randint(0, current_frames_indices.shape[0])
                current_data_indices_mask = [True] * current_number_data_indices + [False] * (current_frames_indices.shape[0] - current_number_data_indices)
                np.random.shuffle(current_data_indices_mask)
                current_person.full_body_data.frames_indices = current_frames_indices[current_data_indices_mask]
                tracks_confidences.append(current_frames_indices[current_data_indices_mask])

        return multiple_persons_tracks, tracks_confidences


    def generate_person_tracked_data(self, frames_indices: np.ndarray, multiple_persons_tracks: MultiplePersonsTracks) -> PersonTrackedData:
        tracked_data = PersonTrackedData()

        video_width = multiple_persons_tracks.video_properties.width
        video_height = multiple_persons_tracks.video_properties.height

        for box_index, frame_index in enumerate(frames_indices):
            current_bounding_box_left = np.random.randint(0, int(video_width / 2))
            current_bounding_box_top = np.random.randint(0, int(video_height / 2))
            current_bounding_box_width = np.random.randint(self.bounding_box_minimal_width, self.video_width - current_bounding_box_left)
            current_bounding_box_height = np.random.randint(self.bounding_box_minimal_height, self.video_height - current_bounding_box_top)
            current_bounding_box = np.array([current_bounding_box_left, current_bounding_box_top, current_bounding_box_width, current_bounding_box_height])

            current_confidence = np.random.random()
            tracked_data.append(current_bounding_box, int(frame_index), current_confidence)

        return tracked_data