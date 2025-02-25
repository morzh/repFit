import copy
import unittest
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.util import regular_grid

from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.filters.single_person.source.filter_addons.bounding_boxes_iou_filter_addon import BoundingBoxesIouFilterAddon
from core.utils.cv.frames_indices import FramesIndices
from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.filters.single_person.source.person_tracked_data import PersonTrackedData
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_properties import VideoProperties
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray


class TestBoundingBoxesIouFilterAddon(unittest.TestCase):

    def setUp(self):
        self.number_checks = 3_500
        self.video_width = 1920
        self.video_height = 1080
        self.minimum_number_of_video_frames = 50
        self.maximum_number_of_video_frames = 5000
        self.minimum_number_persons_in_tracks = 5
        self.maximum_number_persons_in_tracks = 15
        self.maximum_frames_stride_value = 5
        self.joints_number = 17
        self.confidence_range = 0.3, 0.7
        self.show_overall_persons_boxes = False
        self.show_tracks_boxes = False


    def test_bounding_boxes_iou_filter_no_change(self):
        for _ in range(self.number_checks):
            current_persons_number = np.random.randint(self.minimum_number_persons_in_tracks, self.maximum_number_persons_in_tracks + 1)
            current_iou_threshold = np.random.random()
            current_confidence_threshold = np.random.random()
            current_source_tracks = self.generate_tracks_separate_persons(current_persons_number)
            current_filter = BoundingBoxesIouFilterAddon(iou_threshold=current_iou_threshold, confidence_threshold=current_confidence_threshold)
            current_filtered_tracks = copy.deepcopy(current_source_tracks)
            current_filtered_tracks.apply_filter(current_filter)
            self.assertTrue(current_source_tracks.persons == current_filtered_tracks.persons)


    def test_bounding_boxes_iou_filter_confidence(self):
        for _ in range(self.number_checks):
            current_persons_number = np.random.randint(self.minimum_number_persons_in_tracks, self.maximum_number_persons_in_tracks + 1)
            current_iou_threshold = np.random.random()
            current_confidence_threshold = 0.1 + 0.9*np.random.random()
            current_source_tracks = self.generate_tracks_separate_persons(current_persons_number, confidence_range=(current_confidence_threshold + 0.01, 1.0))
            current_source_tracks = self.add_bounding_boxes_to_tracks(current_source_tracks, boxes_confidences_range=(0.01, current_confidence_threshold))
            current_filtered_tracks = copy.deepcopy(current_source_tracks)

            current_filter = BoundingBoxesIouFilterAddon(iou_threshold=current_iou_threshold, confidence_threshold=current_confidence_threshold + 0.01)
            current_filtered_tracks.apply_filter(current_filter)
            self.assertTrue(current_source_tracks.persons == current_filtered_tracks.persons)


    def test_bounding_boxes_iou_filter(self):
        ...


    def generate_tracks_separate_persons(self, persons_number: int, confidence_range=(0.01, 0.99)) -> MultiplePersonsTracks:
        persons_overall_boxes = self.generate_overall_bounding_boxes(persons_number)
        exact_video_frames_number = np.random.randint(self.minimum_number_of_video_frames, self.maximum_number_of_video_frames)
        video_frames_number_inaccuracy = np.random.randint(-10, 10)
        video_frames_stride = np.random.randint(1, self.maximum_frames_stride_value)
        stride_frames_number = int(exact_video_frames_number / video_frames_stride)
        video_frames_indices = np.linspace(0, (exact_video_frames_number // video_frames_stride) * video_frames_stride, stride_frames_number + 1).astype(int)
        approximate_video_frames_number = exact_video_frames_number + video_frames_number_inaccuracy
        fps = 20 + 40 * np.random.random()
        video_properties = VideoProperties('path_to_video', self.video_width, self.video_height, approximate_video_frames_number, fps)
        multiple_persons_tracks = MultiplePersonsTracks(video_properties, stride=video_frames_stride, exact_frames_number=exact_video_frames_number)

        for person_id in range(persons_number):
            current_number_frames_indices = np.random.randint(int(video_frames_indices.shape[0] / 4), int(video_frames_indices.shape[0] / 2))
            current_frames_indices_mask = np.array(current_number_frames_indices * [True] + (video_frames_indices.shape[0] - current_number_frames_indices) * [False])
            np.random.shuffle(current_frames_indices_mask)
            current_frames_indices = video_frames_indices[current_frames_indices_mask]
            multiple_persons_tracks.persons[person_id] = self.generate_single_person_track(current_frames_indices, persons_overall_boxes[person_id], confidence_range)

        if self.show_tracks_boxes:
            self.show_tracks_bounding_boxes(multiple_persons_tracks, persons_overall_boxes)

        return multiple_persons_tracks


    def add_bounding_boxes_to_tracks(self, tracks, boxes_confidences_range=(0.1, 0.5)) -> MultiplePersonsTracks:
        updated_tracks = copy.deepcopy(tracks)
        stride_frames_number = int(tracks.exact_frames_number / tracks.frames_stride)
        frames_indices = np.linspace(0, (tracks.exact_frames_number // tracks.frames_stride) * tracks.frames_stride, stride_frames_number + 1).astype(int)

        for person in updated_tracks.persons.values():
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
                    current_new_confidences = boxes_confidences_range[0] + (boxes_confidences_range[1] - boxes_confidences_range[0]) * np.random.random((current_new_number_of_data_elements,))
                    person.tracked_data.insert(current_new_frames_indices, current_new_bounding_boxes, current_new_confidences)

                    current_new_number_data_frames_indices = np.random.randint(0, np.maximum(current_new_number_of_data_elements, 1))
                    current_data_frames_indices_mask = np.array(
                        current_new_number_data_frames_indices * [True] + (current_new_number_of_data_elements - current_new_number_data_frames_indices) * [False])
                    np.random.shuffle(current_data_frames_indices_mask)
                    current_new_data_indices = current_new_frames_indices[current_data_frames_indices_mask]
                    person.partial_body_data.insert(current_new_data_indices)

                    current_number_full_body_data_frames_indices = np.random.randint(0, np.maximum(current_new_number_of_data_elements, 1))
                    current_full_body_data_frames_indices_mask = np.array(
                        current_number_full_body_data_frames_indices * [True] + (current_new_number_of_data_elements - current_number_full_body_data_frames_indices) * [False])
                    np.random.shuffle(current_full_body_data_frames_indices_mask)
                    current_new_full_body_data_indices = current_new_frames_indices[current_full_body_data_frames_indices_mask]
                    person.full_body_data.insert(current_new_full_body_data_indices)

        return updated_tracks


    def generate_single_person_track(self, tracked_frames_indices, overall_bounding_box: BoundingBox2D, confidences_range=(0.01, 0.99)) -> SinglePersonTrack:
        single_person_track = SinglePersonTrack()
        person_tracked_data = PersonTrackedData()
        indices_number = tracked_frames_indices.shape[0]
        overall_bounding_box_x_bounds = int(overall_bounding_box.x), int(overall_bounding_box.x + overall_bounding_box.width)
        overall_bounding_box_y_bounds = int(overall_bounding_box.y), int(overall_bounding_box.y + overall_bounding_box.height)
        full_body_type_flag = bool(np.random.randint(0, 2))
        bounding_boxes_xs = np.random.randint(overall_bounding_box_x_bounds[0], overall_bounding_box_x_bounds[1], size=(indices_number, 2))
        bounding_boxes_ys = np.random.randint(overall_bounding_box_y_bounds[0], overall_bounding_box_y_bounds[1], size=(indices_number, 2))
        bounding_boxes = np.hstack((bounding_boxes_xs, bounding_boxes_ys))
        bounding_boxes = bounding_boxes[:, [0, 2, 1, 3]]
        bounding_boxes = BoundingBoxes2DArray.xyxy_to_xywh(bounding_boxes)
        bounding_boxes = BoundingBoxes2DArray(bounding_boxes)
        confidences = confidences_range[0] + (confidences_range[1] - confidences_range[0]) * np.random.random((indices_number,))
        person_tracked_data.insert(tracked_frames_indices, bounding_boxes, confidences)
        body_data_reference = single_person_track.full_body_data if full_body_type_flag else single_person_track.partial_body_data
        body_data_reference.frames_indices = FramesIndices(tracked_frames_indices)
        single_person_track.tracked_data = person_tracked_data
        single_person_track.is_active = True
        return single_person_track


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


    def generate_overall_bounding_boxes(self, persons_number: int) -> list[BoundingBox2D]:
        boxes_center_points_random = np.random.random(size=(persons_number, 2))
        boxes_center_points_random[:, 0] *= 0.9 * self.video_width
        boxes_center_points_random[:, 0] += 0.1 * self.video_width
        boxes_center_points_random[:, 1] *= 0.9 * self.video_height
        boxes_center_points_random[:, 1] += 0.1 * self.video_height
        boxes_center_points_random = boxes_center_points_random.astype(np.int64)

        persons_number_square_root = int(np.sqrt(persons_number)) + 1
        boxes_center_points_x = np.linspace(0.05*self.video_width, 0.95*self.video_width, persons_number_square_root)
        boxes_center_points_y = np.linspace(0.05*self.video_height, 0.95*self.video_height, persons_number_square_root)
        boxes_center_points_xs, boxes_center_points_ys = np.meshgrid(boxes_center_points_x, boxes_center_points_y)
        boxes_center_points_regular = np.dstack((boxes_center_points_xs, boxes_center_points_ys)).reshape(-1, 2)
        boxes_center_points_regular_mask = [True] * persons_number + [False] * (boxes_center_points_regular.shape[0] - persons_number)
        boxes_center_points_regular_mask = np.array(boxes_center_points_regular_mask)
        np.random.shuffle(boxes_center_points_regular_mask)
        boxes_center_points_regular = boxes_center_points_regular[boxes_center_points_regular_mask]

        boxes_center_points = 0.15 * boxes_center_points_random + 0.85 * boxes_center_points_regular
        boxes_centers_pairwise_distances_matrix = distance.cdist(boxes_center_points, boxes_center_points, 'euclidean')
        boxes_centers_pairwise_distances_matrix  += np.max(boxes_centers_pairwise_distances_matrix) * np.eye(boxes_centers_pairwise_distances_matrix.shape[0])
        boxes_centers_pairwise_distances_minimum = np.min(boxes_centers_pairwise_distances_matrix)
        boxes_dimensions_value = 0.7 * boxes_centers_pairwise_distances_minimum
        persons_overall_boxes = [BoundingBox2D.from_center_and_dimensions(center, boxes_dimensions_value, boxes_dimensions_value) for center in boxes_center_points]

        if self.show_overall_persons_boxes:
            fig, ax = plt.subplots()
            fig.set_size_inches(22.5, 14.5)
            for box in persons_overall_boxes:
                rect = Rectangle((box.x, box.y), width=box.width, height=box.height, edgecolor='blue', facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)

            plt.axvline(x=0, color=(0, 0, 1, 0.25), label='axvline - full height')
            plt.axvline(x=self.video_width, color=(0, 0, 1, 0.25), label='axvline - full height')

            plt.axhline(y=0, color=(0, 0, 1, 0.25), label='axhline - full width')
            plt.axhline(y=self.video_height, color=(0, 0, 1, 0.25), label='axhline - full width')

            plt.scatter(boxes_center_points[:, 0], boxes_center_points[:, 1], s=2)
            plt.tight_layout()
            plt.show()

        return persons_overall_boxes


    def show_tracks_bounding_boxes(self, tracks:MultiplePersonsTracks, persons_overall_boxes=None) -> None:
        fig, ax = plt.subplots()
        fig.set_size_inches(22.5, 14.5)

        if persons_overall_boxes is not None:
            for box in persons_overall_boxes:
                rect = Rectangle((box.x, box.y), width=box.width, height=box.height, edgecolor='green', facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)

        for person in tracks.persons.values():
            if persons_overall_boxes is None:
                person_overall_box = person.tracked_data.bounding_boxes.circumscribe().flatten()
                rect = Rectangle((person_overall_box[0], person_overall_box[1]), width=person_overall_box[2], height=person_overall_box[3], edgecolor=(0.1, 0.1, 0.85, 0.1), facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)

            for box_index in range(len(person.tracked_data.bounding_boxes)):
                current_box = person.tracked_data.bounding_boxes[box_index].values.flatten()
                rect = Rectangle((current_box[0], current_box[1]), width=current_box[2], height=current_box[3], edgecolor=(0.1, 0.1, 0.85, 0.1), facecolor=(1, 1, 1, 0))
                ax.add_patch(rect)

        plt.axvline(x=0, color=(0, 0, 1, 0.25), label='axvline - full height')
        plt.axvline(x=self.video_width, color=(0, 0, 1, 0.25), label='axvline - full height')

        plt.axhline(y=0, color=(0, 0, 1, 0.25), label='axhline - full width')
        plt.axhline(y=self.video_height, color=(0, 0, 1, 0.25), label='axhline - full width')

        plt.tight_layout()
        plt.show()
