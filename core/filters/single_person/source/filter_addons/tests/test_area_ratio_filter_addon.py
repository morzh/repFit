import copy
import unittest
import numpy as np

from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray

from core.filters.single_person.source.filter_addons.area_ratio_filter_addon import AreaRatioFilterAddon


class TestAreaRatioFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 2_500
        self.video_width = 1920
        self.video_height = 1080
        self.minimum_frames_number = 50
        self.area_ratio_range = 2, 5
        self.area_base_values_range = 150, 500
        self.maximum_number_of_video_frames = 5000
        self.maximum_number_persons_in_tracks = 20
        self.maximum_frames_stride_value = 5
        self.standard_deviation = 1.0
        self.normal_error = 0.3


    def test_area_ratio_filter_addon(self):
        for _ in range(self.number_checks):
            current_area_ratio_threshold = int(self.area_ratio_range[0] + (self.area_ratio_range[1] - self.area_ratio_range[0]) * np.random.rand())
            current_bounding_boxes_area_mean = int(self.area_base_values_range[0] + (self.area_base_values_range[1] - self.area_base_values_range[0]) * np.random.random())

            current_tracks_with_boxes_areas_mean_below_threshold = self.generate_tracks(current_bounding_boxes_area_mean - 1)
            current_tracks_to_filter = self.add_persons_to_tracks(current_tracks_with_boxes_areas_mean_below_threshold, current_bounding_boxes_area_mean * current_area_ratio_threshold)

            current_filter = AreaRatioFilterAddon(area_ratio_threshold=current_area_ratio_threshold)
            current_filter.process(current_tracks_to_filter)

            for person_id in current_tracks_to_filter.persons.keys():
                if person_id in current_tracks_with_boxes_areas_mean_below_threshold.persons.keys():
                    self.assertFalse(current_tracks_to_filter.persons[person_id].is_active)
                else:
                    self.assertTrue(current_tracks_to_filter.persons[person_id].is_active)

            # self.assertTrue(current_tracks_with_boxes_areas_mean_below_threshold == current_tracks_to_filter)


    def generate_tracks(self, bounding_boxes_area_mean: int) -> MultiplePersonsTracks:
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

            current_person_tracked_data = self.generate_person_tracked_data(current_frames_indices, bounding_boxes_area_mean, multiple_persons_tracks)
            current_person = SinglePersonTrack()
            current_person.tracked_data = current_person_tracked_data
            multiple_persons_tracks.persons[person_index] = current_person

        return multiple_persons_tracks


    def generate_person_tracked_data(self, frames_indices: np.ndarray, bounding_boxes_mean: np.integer, tracks: MultiplePersonsTracks) -> PersonTrackedData:
        tracked_data = PersonTrackedData()

        for box_index, frame_index in enumerate(frames_indices):
            current_bounding_box_left = np.random.randint(0, int(tracks.video_properties.width / 2))
            current_bounding_box_top = np.random.randint(0, int(tracks.video_properties.height / 2))
            current_bounding_box_width = round(np.random.normal(bounding_boxes_mean, self.standard_deviation))
            current_bounding_box_width = max(int(current_bounding_box_width), 1)
            current_bounding_box_height = 1
            current_bounding_box = np.array([current_bounding_box_left, current_bounding_box_top, current_bounding_box_width, current_bounding_box_height])

            current_confidence = np.random.random()

            tracked_data.append(current_bounding_box, int(frame_index), current_confidence)

        return tracked_data


    def add_persons_to_tracks(self, tracks, bounding_boxes_area_mean) -> MultiplePersonsTracks:
        stride_frames_number = int(tracks.exact_frames_number / tracks.frames_stride)
        frames_indices = np.linspace(0, (tracks.exact_frames_number // tracks.frames_stride) * tracks.frames_stride, stride_frames_number + 1).astype(int)
        new_persons_number = np.random.randint(0, 20)

        new_persons_starting_index = len(tracks.persons)
        for person_index in range(new_persons_starting_index, new_persons_starting_index + new_persons_number):
            current_person_tracked_data = PersonTrackedData()
            current_number_tracked_frames_indices = np.random.randint(1, frames_indices.shape[0])
            current_tracked_frames_indices_mask = np.array(current_number_tracked_frames_indices * [True] + (frames_indices.shape[0] - current_number_tracked_frames_indices) * [False])
            np.random.shuffle(current_tracked_frames_indices_mask)

            current_tracked_frames_indices = frames_indices[current_tracked_frames_indices_mask]
            current_tracked_bounding_boxes = self.generate_bounding_boxes(current_number_tracked_frames_indices, tracks.video_properties, bounding_boxes_area_mean)
            current_tracked_confidences = np.random.random((current_number_tracked_frames_indices,))
            current_person_tracked_data.insert(current_tracked_frames_indices, current_tracked_bounding_boxes, current_tracked_confidences)

            current_person_track = SinglePersonTrack()
            current_person_track.tracked_data = current_person_tracked_data
            tracks.persons[person_index] = current_person_track

        return tracks


    def generate_bounding_boxes(self, number_boxes: int, video_properties: VideoProperties, area_mean: np.number) -> BoundingBoxes2DArray:
        size = (number_boxes, 1)
        lefts = np.random.randint(0, video_properties.width, size)
        tops = np.random.randint(0, video_properties.height, size)
        widths = np.random.normal(area_mean, self.standard_deviation, size=size)
        heights = np.ones(size)

        boxes = np.hstack((lefts, tops, widths, heights))
        bounding_boxes = BoundingBoxes2DArray(boxes)
        return bounding_boxes
