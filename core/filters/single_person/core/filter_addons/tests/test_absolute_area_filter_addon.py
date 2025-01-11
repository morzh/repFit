import unittest
import numpy as np

from core.filters.single_person.core.filter_addons.absolute_area_filter_addon import AbsoluteAreaFilterAddon
from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.core.person_tracked_data import PersonTrackedData
from core.filters.single_person.core.single_person_track import SinglePersonTrack
from core.utils.cv.video_properties import VideoProperties


class TestAbsoluteAreaFilterAddon(unittest.TestCase):

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


    def test_area_filter(self):
        for _ in range(self.number_checks):
            current_persons_number = np.random.randint(1, self.number_persons_in_tracks)
            current_exact_frames_number = np.random.randint(self.minimum_frames_number, self.maximum_number_of_video_frames)
            current_stride = np.random.randint(1, self.maximum_frames_stride_value)
            current_number_bounding_boxes_areas = np.random.randint(1, int(current_exact_frames_number / current_stride))
            current_bounding_boxes_areas = [np.random.randint(self.minimum_box_area, self.maximum_box_area, current_number_bounding_boxes_areas) for _ in range(current_persons_number)]
            current_absolute_area_threshold = np.percentile(np.array(current_bounding_boxes_areas), 50.0)
            current_tracks = self.generate_tracks(self.video_width, self.video_height, current_exact_frames_number, current_stride, current_bounding_boxes_areas)

            current_boxes_mean = np.mean(np.array(current_bounding_boxes_areas), axis=1)
            current_boxes_mean_mask = current_boxes_mean >= current_absolute_area_threshold

            current_filter = AbsoluteAreaFilterAddon(area_threshold=current_absolute_area_threshold)
            current_filter.process(current_tracks, filter_full_body_person=False)

            for person_id, person_track in current_tracks.persons.items():
                self.assertTrue(bool(len(person_track.data.frames_indices)) == current_boxes_mean_mask[person_id])


    def generate_tracks(self, video_width: int, video_height: int, exact_video_frames_number: int, video_frames_stride: int, bounding_boxes_areas: list[np.ndarray]) -> MultiplePersonsTracks:
        stride_frames_number = int(exact_video_frames_number / video_frames_stride)
        frames_indices = np.linspace(0, (exact_video_frames_number // video_frames_stride) * video_frames_stride, stride_frames_number + 1).astype(int)
        persons_number = len(bounding_boxes_areas)
        video_frames_number_inaccuracy = np.random.randint(-10, 10)
        approximate_video_frames_number = exact_video_frames_number + video_frames_number_inaccuracy
        fps = 20 + 40*np.random.random()

        video_properties = VideoProperties('path_to_video', video_width, video_height, approximate_video_frames_number, fps)
        multiple_persons_tracks = MultiplePersonsTracks(video_properties, stride=video_frames_stride, exact_frames_number=exact_video_frames_number)

        for person_index in range(persons_number):
            current_person_tracked_data = self.generate_person_tracked_data(frames_indices, bounding_boxes_areas[person_index], multiple_persons_tracks)
            current_person = SinglePersonTrack()
            current_person.tracked_data = current_person_tracked_data
            multiple_persons_tracks.persons[person_index] = current_person

        return multiple_persons_tracks


    def generate_person_tracked_data(self, frames_indices: np.ndarray, bounding_boxes_areas: np.ndarray, multiple_persons_tracks: MultiplePersonsTracks) -> PersonTrackedData:
        tracked_data = PersonTrackedData()
        number_bounding_boxes = bounding_boxes_areas.shape[0]
        number_frames_indices = frames_indices.shape[0]
        frames_indices_mask = np.array(number_bounding_boxes * [True] + (number_frames_indices - number_bounding_boxes) * [False])
        np.random.shuffle(frames_indices_mask)

        masked_frames_indices = frames_indices[frames_indices_mask]
        video_width = multiple_persons_tracks.video_properties.width
        video_height = multiple_persons_tracks.video_properties.height

        for box_index, frame_index in enumerate(masked_frames_indices):
            current_bounding_box_left = np.random.randint(0, int(video_width / 2))
            current_bounding_box_top = np.random.randint(0, int(video_height / 2))
            '''
            current_bounding_boxes_areas_divisors = self.find_divisors(int(bounding_boxes_areas[box_index]))
            current_bounding_box_width = current_bounding_boxes_areas_divisors[np.random.randint(0, current_bounding_boxes_areas_divisors.shape[0] - 1)]
            current_bounding_box_height = int(bounding_boxes_areas[box_index] / current_bounding_box_width)
            '''
            current_bounding_box_width = bounding_boxes_areas[box_index]
            current_bounding_box_height = 1
            current_bounding_box = np.array([current_bounding_box_left, current_bounding_box_top, current_bounding_box_width, current_bounding_box_height])

            current_confidence = np.random.random()

            current_joints_x_coordinates = np.random.randint(current_bounding_box_left, current_bounding_box_left + current_bounding_box_width, (self.joints_number, )).reshape(-1, 1)
            current_joints_y_coordinates = np.random.randint(current_bounding_box_top, current_bounding_box_top + current_bounding_box_height, (self.joints_number, )).reshape(-1, 1)
            current_joints_confidences = np.random.random(self.joints_number).reshape(-1, 1)
            current_joints  = np.hstack((current_joints_x_coordinates, current_joints_y_coordinates, current_joints_confidences))

            tracked_data.append(current_bounding_box, int(frame_index), current_confidence, joints=current_joints)

        return tracked_data


    @staticmethod
    def find_divisors(value: int) -> np.ndarray:
        divisors = []
        for i in range(1, value + 1):
            if value % i == 0:
                divisors.append(i)
        return np.array(divisors)