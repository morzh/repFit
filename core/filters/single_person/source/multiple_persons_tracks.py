import cv2
import numpy as np
import os
import pickle

from fiftyone.core.frame import Frames

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.single_person.source.frames_indices import FramesIndices
from core.utils.cv.frames_segments import FramesSegments
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_reader import VideoReader
from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray
import core.utils.visualization.tracks_visualizing_utils as viz


class MultiplePersonsTracks:
    """
    Description:
        Persons data storage class.

    :ivar persons: person_id -> person data mapping
    :ivar video_properties:  video properties data
    """
    def __init__(self, video_properties: VideoProperties, stride=1, exact_frames_number=-1):
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties
        self.exact_frames_number = exact_frames_number
        self._frames_stride = stride


    def __eq__(self, other):
        return (self.video_properties == other.video_properties and
                self.exact_frames_number == other.exact_frames_number and
                self.frames_stride == other.frames_stride and
                self.persons == other.persons)


    def update(self, frame_index: int, bounding_boxes: np.ndarray, keypoints: np.ndarray | None = None) -> None:
        """
        Description:
            Update persons tracks data.

        :param bounding_boxes: person's data, obtained from AI model;
        :param frame_index: frame index.
        :param keypoints: tracked keypoints
        """
        number_persons = bounding_boxes.shape[0]
        for index in range(number_persons):
            current_person_id = int(bounding_boxes[index, 4])

            if current_person_id == 0:
                """ Case if no person ids tracked """
                continue
            elif current_person_id not in self.persons:
                self.persons[current_person_id] = SinglePersonTrack()

            current_confidence = float(bounding_boxes[index, 5])
            current_bounding_box = bounding_boxes[index, :4]
            current_joints = keypoints[index] if keypoints is not None else None

            self.persons[current_person_id].tracked_data.append(current_bounding_box, frame_index, current_confidence, joints=current_joints, bounding_box_mode=BoundingBoxes2DArray.XYXY)


    def apply_filter(self, filter_visitor: MultiPersonsFilterAddonBase, apply_to_full_body=False) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        :param apply_to_full_body: apply filter to full body data
        """
        filter_visitor.process(self, filter_full_body_person=apply_to_full_body)


    def serialize(self, filepath: os.PathLike | str) -> None:
        """
        Description:
            Serialize class instance.

        :param filepath: filepath to write class instance to.
        """
        with open(filepath, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    def clip_persons_segments(self, person_segments: FramesSegments, active_person_id: int) -> dict[int, FramesSegments]:
        """
        Description:

        :param person_segments:
        :param active_person_id:

        :return:
        """
        clipped_persons_segments = {}
        for person_id, person_track in self.persons.items():
            if person_id == active_person_id: continue
            current_clipped_frame_segments = person_track.data.clip_segments(person_segments)
            clipped_persons_segments[person_id] = current_clipped_frame_segments

        return clipped_persons_segments


    def persons_bounding_boxes(self, persons_segments: dict[int, FramesSegments], active_person_id: int) -> dict[int, BoundingBoxes2DArray]:
        """
        Description:
            Bounding boxes per person ID except  of person with ``active_person_id``.

        :param persons_segments: per person frame segments;
        :param active_person_id: person id, excluded from result.

        :return: bounding boxes array per person id.
        """
        bounding_boxes = {}
        for person_id, person_track in self.persons.items():
            if person_id == active_person_id: continue
            bounding_boxes[person_id] = person_track.bounding_boxes_per_segment(persons_segments[person_id])

        return  bounding_boxes


    def visualize_tracked_data(self, **options) -> None:
        """
        Description:
            Visualize multiple persons tracks.

        :keyword frames_thickness: person bounding box thickness
        :keyword show_frame_delay: gap between gaps in milliseconds
        :keyword hsv_step: HSV format hue circle step (in degrees)
        """
        def calculate_segments(frames_indices: FramesIndices, stride=1) -> FramesSegments:
            """
            Description:
                Calculate frames segments using  video  frames``stride`` value.

            :param frames_indices: video frames indices;
            :param stride: video frames stride
            """
            segments_bins = np.hstack((frames_indices.values.reshape(-1, 1), frames_indices.values.reshape(-1, 1) + stride))
            for index in range(segments_bins.shape[0] - 1):
                if segments_bins[index, 1] == segments_bins[index + 1, 0]:
                    segments_bins[index + 1, 0] = segments_bins[index, 0]
                    segments_bins[index] = -1

            mask = segments_bins[:, 0] != -1
            segments = segments_bins[mask]
            segments[:, 1] += 1
            return FramesSegments(segments)

        boxes_thickness = options.get('frames_thickness', 2)
        show_frame_delay = options.get('show_frame_delay', 10)
        hsv_step = options.get('hsv_step', 20)

        video_reader = VideoReader(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)
        imshow_window_name = video_filename  # .encode('ascii', 'ignore')

        for frame in video_reader:
            for person_id, person_track in self.persons.items():
                mean_boxes_area = self.persons[person_id].mean_height()
                joints_radius = max(int(round(mean_boxes_area * 0.01)), 1)
                bones_thickness = max(int(round(joints_radius / 2)), 1)
                current_segments = calculate_segments(person_track.tracked_data.frames_indices, self.frames_stride)

                for segment in current_segments:
                    if segment[0] <= video_reader.current_frame_index < segment[1]:
                        current_bounding_box = self.persons[person_id].tracked_data.bounding_box(video_reader.current_frame_index)
                        current_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_bounding_box.astype(np.int64))[0]
                        current_keypoints = self.persons[person_id].tracked_data.frame_keypoints(video_reader.current_frame_index)
                        current_confidence = self.persons[person_id].tracked_data.confidence(video_reader.current_frame_index)
                        current_color = viz.stepped_color(hsv_step, person_id)
                        current_boxes_overlay = viz.draw_bounding_boxes(person_id, frame, current_bounding_box, current_color, boxes_thickness)

                        frame = cv2.addWeighted(current_boxes_overlay, current_confidence, frame, 1 - current_confidence, 0)
                        if person_track.tracked_data.joints is not None:
                            frame = viz.draw_skeleton_joints(frame, current_color, joints_radius, current_keypoints)
                            frame = viz.draw_skeleton_bones(frame, current_color, bones_thickness, current_keypoints)

            cv2.imshow(imshow_window_name, frame)
            cv2.waitKey(show_frame_delay)

        cv2.destroyAllWindows()


    @property
    def frames_stride(self):
        return self._frames_stride


    @staticmethod
    def enlarge_bounding_boxes(boxes_to_enlarge, obstacles_boxes, borderline_bounding_box: BoundingBox2D) -> BoundingBoxes2DArray:
        """
        Description:
            Enlarge ``boxes_to_enlarge`` bounding boxes to the borders of ``obstacles_boxes`` and ``borderline_bounding_box``.

        :param boxes_to_enlarge: bounding boxes to enlarge;
        :param obstacles_boxes: obstacles bounding boxes;
        :param borderline_bounding_box: borderline (maximum possible) bounding box.

        :return: enlarged bounding boxes array
        """
        for box in boxes_to_enlarge:
            current_box = BoundingBox2D.from_numpy(box)
            current_box = current_box.enlarge(obstacles_boxes, borderline_bounding_box)
            box.values = current_box.to_numpy()
        return boxes_to_enlarge


    @staticmethod
    def intersections_over_unions(active_person_boxes: BoundingBoxes2DArray, other_persons_boxes: dict[int, BoundingBoxes2DArray]) -> dict[int, list[np.ndarray]]:
        """
        Description:

        :param active_person_boxes:
        :param other_persons_boxes:

        :return:
        """
        result_intersection_areas = {}
        for person_id, person_boxes in other_persons_boxes.items():
            current_intersection_boxes = BoundingBoxes2DArray.intersect(active_person_boxes, person_boxes)
            current_intersection_areas = [boxes.areas() for boxes in current_intersection_boxes if len(boxes)]
            result_intersection_areas[person_id] = current_intersection_areas if len(current_intersection_areas) else np.zeros(1,)

        return result_intersection_areas
