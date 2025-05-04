import copy
import unittest
import numpy as np

from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.filters.single_person.source.filter_addons.joints_filter_addon import JointsFilterAddon

from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class TestJointsFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 1_500
        self.video_width = 1_920
        self.video_height = 1_080
        self.minimum_frames_number = 50
        self.minimum_box_area = 300
        self.maximum_box_area = int(0.5 * self.video_width * self.video_height)
        self.maximum_number_of_video_frames = 5_000
        self.maximum_number_persons_in_tracks = 20
        self.maximum_frames_stride_value = 5
        self.maximum_area_threshold = 1_200
        self.bounding_box_minimal_width = 30
        self.bounding_box_minimal_height = 30
        self.joints_number = 17
        self.confidence_range = 0.3, 0.7
        # self.useful_joints_number = 16


    def test_joints_filter_edge_cases_set_1(self):
        for _ in range(self.number_checks):
            current_confidence_threshold_value = self.confidence_range[0] + (self.confidence_range[1] - self.confidence_range[0]) * np.random.rand()
            current_number_joints_threshold = np.random.randint(1, 16)

            current_tracks = self.generate_tracks(confident_joints_number_range=(current_number_joints_threshold, self.joints_number),
                                                  confident_joints_confidences_range=(current_confidence_threshold_value, 1.0))

            current_full_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                         minimum_number_joints_threshold=current_number_joints_threshold,
                                                         maximum_number_joints_threshold=self.joints_number,
                                                         filter_full_body_person=True)
            current_full_body_filter.process(current_tracks)
            for person in current_tracks.persons.values():
                self.assertTrue(np.all(person.tracked_data.frames_indices == person.full_body_data.frames_indices))

            current_partial_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                            minimum_number_joints_threshold=current_number_joints_threshold,
                                                            maximum_number_joints_threshold=self.joints_number,
                                                            filter_full_body_person=False)
            current_partial_body_filter.process(current_tracks)
            for person in current_tracks.persons.values():
                self.assertTrue(np.all(person.tracked_data.frames_indices == person.partial_body_data.frames_indices))
            # -----------------------------------------------------------------------------------------------------------

            current_full_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                         minimum_number_joints_threshold=0,
                                                         maximum_number_joints_threshold=current_number_joints_threshold - 1,
                                                         filter_full_body_person=True)
            current_full_body_filter.process(current_tracks)
            for person in current_tracks.persons.values():
                self.assertTrue(len(person.full_body_data.frames_indices) == 0)

            current_partial_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                            minimum_number_joints_threshold=0,
                                                            maximum_number_joints_threshold=current_number_joints_threshold - 1,
                                                            filter_full_body_person=False)
            current_partial_body_filter.process(current_tracks)
            for person in current_tracks.persons.values():
                self.assertTrue(len(person.partial_body_data.frames_indices) == 0)


    def test_joints_filter_edge_cases_set_2(self):
        for _ in range(self.number_checks):
            current_confidence_threshold_value = self.confidence_range[0] + (self.confidence_range[1] - self.confidence_range[0]) * np.random.rand()
            current_number_joints_threshold = np.random.randint(1, 16)

            current_tracks = self.generate_tracks(confident_joints_number_range=(current_number_joints_threshold, self.joints_number),
                                                  confident_joints_confidences_range=(self.confidence_range[0] - 2e-6, current_confidence_threshold_value - 1e-6))

            current_full_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                         minimum_number_joints_threshold=current_number_joints_threshold,
                                                         maximum_number_joints_threshold=self.joints_number,
                                                         filter_full_body_person=True)
            current_full_body_filter.process(current_tracks)
            for person in current_tracks.persons.values():
                self.assertTrue(len(person.full_body_data.frames_indices) == 0)


    def test_joints_filter_non_edge_cases(self):
        for _ in range(self.number_checks):
            current_confidence_threshold_value = self.confidence_range[0] + (self.confidence_range[1] - self.confidence_range[0]) * np.random.rand()
            current_minimum_number_joints_low_threshold = np.random.randint(1, self.joints_number - 1)
            current_minimum_number_joints_high_threshold = np.minimum(current_minimum_number_joints_low_threshold + np.random.randint(1, self.joints_number), self.joints_number)

            current_tracks = self.generate_tracks(confident_joints_number_range=(current_minimum_number_joints_low_threshold, current_minimum_number_joints_high_threshold),
                                                  confident_joints_confidences_range=(current_confidence_threshold_value, 1.0))

            current_modified_tracks = self.add_data_to_tracks(current_tracks,
                                                              confident_joints_number_range=(0, current_minimum_number_joints_low_threshold - 1),
                                                              confident_joints_confidences_range=(current_confidence_threshold_value, 1.0))

            if current_minimum_number_joints_high_threshold < self.joints_number:
                current_modified_tracks = self.add_data_to_tracks(current_modified_tracks,
                                                                  confident_joints_number_range=(current_minimum_number_joints_high_threshold + 1, self.joints_number),
                                                                  confident_joints_confidences_range=(current_confidence_threshold_value, 1.0))

            current_partial_body_filter = JointsFilterAddon(joints_confidence_threshold=current_confidence_threshold_value,
                                                            minimum_number_joints_threshold=current_minimum_number_joints_low_threshold,
                                                            maximum_number_joints_threshold=current_minimum_number_joints_high_threshold,
                                                            filter_full_body_person=False)
            current_partial_body_filter.process(current_modified_tracks)
            for person_id, person_track in current_tracks.persons.items():
                self.assertTrue(np.all(person_track.tracked_data.frames_indices == current_modified_tracks.persons[person_id].partial_body_data.frames_indices))


    def generate_tracks(self, confident_joints_number_range=(10, 17), confident_joints_confidences_range=(0.1, 0.95)) -> MultiplePersonsTracks:
        persons_number = np.random.randint(1, self.maximum_number_persons_in_tracks)
        exact_video_frames_number = np.random.randint(self.minimum_frames_number, self.maximum_number_of_video_frames)
        video_frames_stride = np.random.randint(1, self.maximum_frames_stride_value)
        stride_frames_number = int(exact_video_frames_number / video_frames_stride)
        video_frames_indices = np.linspace(0, (exact_video_frames_number // video_frames_stride) * video_frames_stride, stride_frames_number + 1).astype(int)
        video_frames_number_inaccuracy = np.random.randint(-10, 10)
        approximate_video_frames_number = exact_video_frames_number + video_frames_number_inaccuracy
        fps = 20 + 40*np.random.random()

        video_properties = VideoProperties('path_to_video', self.video_width, self.video_height, approximate_video_frames_number, fps)
        multiple_persons_tracks = MultiplePersonsTracks(video_properties, stride=video_frames_stride, exact_frames_number=exact_video_frames_number)

        for person_index in range(persons_number):
            current_number_frames_indices = np.random.randint(int(video_frames_indices.shape[0] / 4), int(video_frames_indices.shape[0] / 2))
            current_frames_indices_mask = np.array(current_number_frames_indices * [True] + (video_frames_indices.shape[0] - current_number_frames_indices) * [False])
            np.random.shuffle(current_frames_indices_mask)
            current_frames_indices = video_frames_indices[current_frames_indices_mask]

            current_person_tracked_data = self.generate_person_tracked_data(current_frames_indices, confident_joints_number_range, confident_joints_confidences_range, multiple_persons_tracks)
            current_person = SinglePersonTrack()
            current_person.tracked_data = current_person_tracked_data
            current_person.is_active = True
            multiple_persons_tracks.persons[person_index] = current_person

        return multiple_persons_tracks


    def generate_person_tracked_data(self, frames_indices: np.ndarray, joints_number_range: tuple, confident_joints_confidence_range: tuple, tracks: MultiplePersonsTracks) -> PersonTrackedData:
        tracked_data = PersonTrackedData()

        for box_index, frame_index in enumerate(frames_indices):
            current_bounding_box_left = np.random.randint(0, int(tracks.video_properties.width / 2))
            current_bounding_box_top = np.random.randint(0, int(tracks.video_properties.height / 2))
            current_bounding_box_width = np.random.randint(self.bounding_box_minimal_width, self.video_width - current_bounding_box_left)
            current_bounding_box_height = np.random.randint(self.bounding_box_minimal_height, self.video_height - current_bounding_box_top)
            current_bounding_box = np.array([current_bounding_box_left, current_bounding_box_top, current_bounding_box_width, current_bounding_box_height])

            current_confidence = np.random.random()

            joints_minimum = np.ones(2,)
            joints_range = np.array((current_bounding_box_width, current_bounding_box_height))
            current_number_joints = np.random.randint(joints_number_range[0], joints_number_range[1] + 1)
            current_joints_positions = joints_minimum + joints_range * np.random.random((self.joints_number, 2))
            current_joints_in_range_confidences = confident_joints_confidence_range[0] + (confident_joints_confidence_range[1] - confident_joints_confidence_range[0]) * np.random.random(current_number_joints, )
            if current_number_joints < self.joints_number:
                current_joints_out_of_range_confidences = (confident_joints_confidence_range[0] - 1e-6) * np.random.random(self.joints_number - current_number_joints)
                current_joints_confidences = np.append(current_joints_in_range_confidences, current_joints_out_of_range_confidences)
            else:
                current_joints_confidences = current_joints_in_range_confidences
            np.random.shuffle(current_joints_confidences)
            current_joints = np.hstack((current_joints_positions, current_joints_confidences.reshape(-1, 1)))

            tracked_data.append(current_bounding_box, int(frame_index), current_confidence, current_joints)

        return tracked_data


    def add_data_to_tracks(self, tracks: MultiplePersonsTracks, confident_joints_number_range=(10, 17), confident_joints_confidences_range=(0.1, 0.95)) -> MultiplePersonsTracks:
        stride_frames_number = int(tracks.exact_frames_number / tracks.frames_stride)
        frames_indices = np.linspace(0, (tracks.exact_frames_number // tracks.frames_stride) * tracks.frames_stride, stride_frames_number + 1).astype(int)

        new_tracks = copy.deepcopy(tracks)
        minimum_video_dimension = min(tracks.video_properties.width, tracks.video_properties.height)

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
                    current_new_confidences = np.random.random((current_new_number_of_data_elements,))
                    current_new_joints_positions = np.random.randint(0, minimum_video_dimension, (current_new_number_of_data_elements, 17, 2))
                    current_new_number_useful_joints = np.random.randint(confident_joints_number_range[0], confident_joints_number_range[1] + 1)
                    current_new_joints_usable_confidences = (confident_joints_confidences_range[0] + (confident_joints_confidences_range[1] - confident_joints_confidences_range[0]) *
                                                             np.random.random((current_new_number_of_data_elements, current_new_number_useful_joints, 1)))
                    current_new_joints_non_usable_confidences = ((confident_joints_confidences_range[0] - 1e-6) *
                                                                 np.random.random((current_new_number_of_data_elements, self.joints_number - current_new_number_useful_joints, 1)))
                    current_new_joints_confidences = np.hstack((current_new_joints_usable_confidences, current_new_joints_non_usable_confidences))
                    np.random.shuffle(current_new_joints_confidences)
                    current_new_joints = np.dstack((current_new_joints_positions, current_new_joints_confidences))
                    person.tracked_data.insert(current_new_frames_indices, current_new_bounding_boxes, current_new_confidences, current_new_joints)

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
