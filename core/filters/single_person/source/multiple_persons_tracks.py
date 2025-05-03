import cv2
import numpy as np
import os
import pickle

from core.filters.single_person.source.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.filters.video_filter.cut_steady_video import video_seg
from core.utils.cv.frames_segments import FramesSegments
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.source.single_person_track import SinglePersonTrack
from core.utils.cv.video_reader import VideoReader
from core.utils.geometry.bounding_boxes.bounding_box_2d import BoundingBox2D
from core.utils.geometry.bounding_boxes.bounding_boxes_2d_array import BoundingBoxes2DArray
import core.utils.visualization.tracks_visualizing_utils as viz
from core.utils.geometry.segments.segment_2d import Segment2D


class MultiplePersonsTracks:
    """
    Description:
        Persons data storage class.

    :ivar persons: person_id -> person data mapping
    :ivar video_properties:  video properties data
    """
    def __init__(self, video_properties: VideoProperties, stride=1, exact_frames_number=-1):
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties: VideoProperties = video_properties
        self.exact_frames_number: int = exact_frames_number
        self._frames_stride: int = stride
        self.filtering_chain: list[MultiPersonsFilterAddonBase] = []


    def __eq__(self, other):
        return (self.video_properties == other.video_properties and
                self.exact_frames_number == other.exact_frames_number and
                self.frames_stride == other.frames_stride and
                self.persons == other.persons and
                self.filtering_chain == other.filtering_chain)


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
            self.persons[current_person_id].is_active = True


    def apply_filter(self, filter_visitor: MultiPersonsFilterAddonBase) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        """
        self.filtering_chain.append(filter_visitor)
        filter_visitor.process(self)


    def clear_filtering_chain(self) -> None:
        """
        Description:
            Clear filters list, applied to tracks.
        """
        self.filtering_chain = []


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
            current_clipped_frame_segments = person_track.partial_body_data.clip_segments(person_segments)
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


    def visualize_tracked_data(self, **parameters) -> None:
        """
        Description:
            Visualize tracked data.

        :keyword frames_thickness: person bounding box thickness;
        :keyword show_frame_delay: delay in milliseconds between frames (for cv2.imshow() function);
        :keyword hsv_step: HSV format hue circle step (in degrees);
        :keyword skip_existing_videos: if ``True``, skip video files that already exist in ``visualization_videos_folder``;
        :keyword visualization_videos_folder: folder to write visualized videos to;
        :keyword maximum_width: maximum width of visualized videos;
        :keyword maximum_height: maximum height of visualized videos;
        :keyword write_video: if ``True``, write video file;
        :keyword show_frames: if ``True``, show visualized frames.
        """
        skip_existing_videos = parameters.get('skip_existing_videos', True)
        videos_folder = os.path.dirname(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)

        video_visualized_folder = os.path.join(videos_folder, parameters['visualization_videos_folder'])
        os.makedirs(os.path.dirname(video_visualized_folder), exist_ok=True)
        video_visualized_filepath = os.path.join(str(video_visualized_folder), video_filename + '.viz.mp4')

        if skip_existing_videos and os.path.exists(video_visualized_filepath): return

        boxes_thickness = parameters.get('frames_thickness', 2)
        show_frame_delay = parameters.get('show_frame_delay', 10)
        hsv_step = parameters.get('hsv_step', 20)
        maximum_width = parameters.get('maximum_width', 1920)
        maximum_height = parameters.get('maximum_height', 1080)
        write_video = parameters.get('write_video', False)
        show_frames = parameters.get('show_frames', True)

        video_reader = VideoReader(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)
        video_width_height = (video_reader.video_properties.width, video_reader.video_properties.height)

        if video_reader.video_properties.width > maximum_width or video_reader.video_properties.height > maximum_height:
            factor = min(maximum_height / video_reader.video_properties.height, maximum_width / video_reader.video_properties.width)
            video_width_height = (int(factor * video_reader.video_properties.width), int(factor * video_reader.video_properties.height))

        if write_video:
            video_writer = cv2.VideoWriter(video_visualized_filepath, cv2.VideoWriter_fourcc(*'MP4V'), video_reader.video_properties.fps, video_width_height)
        else:
            video_writer = None

        for frame in video_reader:
            for person_id, person_track in self.persons.items():
                current_mean_boxes_area = person_track.mean_height()
                current_joints_radius = max(int(round(current_mean_boxes_area * 0.01)), 1)
                current_bones_thickness = max(int(round(current_joints_radius / 2)), 1)
                current_person_color = viz.stepped_color(hsv_step, person_id) if person_track.is_active else (120, 120, 120, 120)

                if person_track.tracked_data.frames_indices.is_in_vicinity(video_reader.current_frame_index, self.frames_stride, mode='right'):
                    current_bounding_box = person_track.tracked_data.bounding_box(video_reader.current_frame_index)
                    current_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_bounding_box.values.astype(np.int64))[0]
                    current_keypoints = person_track.tracked_data.frame_keypoints(video_reader.current_frame_index)
                    current_confidence = person_track.tracked_data.confidence(video_reader.current_frame_index)

                    current_boxes_overlay = viz.draw_bounding_boxes(person_id, frame, current_bounding_box, current_person_color, boxes_thickness)
                    frame = cv2.addWeighted(current_boxes_overlay, current_confidence, frame, 1 - current_confidence, 0)
                    if person_track.tracked_data.joints is not None:
                        frame = viz.draw_skeleton_joints(frame, current_person_color, current_joints_radius, current_keypoints)
                        frame = viz.draw_skeleton_bones(frame, current_person_color, current_bones_thickness, current_keypoints)

            frame = cv2.resize(frame, video_width_height)
            if write_video: video_writer.write(frame)
            if show_frames:
                cv2.imshow(video_filename, frame)
                cv2.waitKey(show_frame_delay)

        if write_video: video_writer.release()
        if show_frames: cv2.destroyAllWindows()


    def visualize_full_body_data(self, **parameters) -> None:
        """
        Description:
            Visualize full body data of persons tracks.

        :keyword frames_thickness: person bounding box thickness
        :keyword show_frame_delay: delay in milliseconds between frames (for cv2.imshow() function)
        :keyword hsv_step: HSV format hue circle step (in degrees)
        :keyword skip_existing_videos: if ``True``, skip video files that already exist in ``visualization_videos_folder``
        :keyword visualization_videos_folder: folder to write visualized videos to
        :keyword maximum_width: maximum width of visualized videos
        :keyword maximum_height: maximum height of visualized videos
        :keyword write_video: if ``True``, write video file
        :keyword show_frames: if ``True``, show visualized frames
        """
        skip_existing_videos = parameters.get('skip_existing_videos', True)
        videos_folder = os.path.dirname(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)

        video_visualized_folder = os.path.join(videos_folder, parameters['visualization_videos_folder'])
        os.makedirs(os.path.dirname(video_visualized_folder), exist_ok=True)
        video_visualized_filepath = os.path.join(str(video_visualized_folder), video_filename + '.viz.mp4')

        if skip_existing_videos and os.path.exists(video_visualized_filepath): return

        boxes_thickness = parameters.get('frames_thickness', 2)
        show_frame_delay = parameters.get('show_frame_delay', 10)
        hsv_step = parameters.get('hsv_step', 20)
        maximum_width = parameters.get('maximum_width', 1920)
        maximum_height = parameters.get('maximum_height', 1080)
        write_video = parameters.get('write_video', False)
        show_frames = parameters.get('show_frames', True)

        video_reader = VideoReader(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)
        video_width_height = (video_reader.video_properties.width, video_reader.video_properties.height)

        if video_reader.video_properties.width > maximum_width or video_reader.video_properties.height > maximum_height:
            factor = min(maximum_height / video_reader.video_properties.height, maximum_width / video_reader.video_properties.width)
            video_width_height = (int(factor * video_reader.video_properties.width), int(factor * video_reader.video_properties.height))

        if write_video:
            video_writer = cv2.VideoWriter(video_visualized_filepath, cv2.VideoWriter_fourcc(*'MP4V'), video_reader.video_properties.fps, video_width_height)
        else:
            video_writer = None

        for frame_index, frame in enumerate(video_reader):
            for person_id, person_track in self.persons.items():
                current_mean_boxes_area = person_track.mean_height()
                current_joints_radius = max(int(round(current_mean_boxes_area * 0.01)), 1)
                current_bones_thickness = max(int(round(current_joints_radius / 2)), 1)
                current_person_color = viz.stepped_color(hsv_step, person_id) if person_track.is_active else (120, 120, 120, 120)

                if len(person_track.full_body_data.segments):
                    current_segments_bounding_boxes = person_track.full_body_data.segments.bounding_boxes()
                    if frame_index in person_track.full_body_data.frames_segments:
                        current_segment_index = person_track.full_body_data.frames_segments.index(frame_index)
                        current_confidence = person_track.full_body_data.confidence(video_reader.current_frame_index)
                        current_keypoints = person_track.full_body_data.frame_keypoints(video_reader.current_frame_index)
                        current_overall_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_segments_bounding_boxes[current_segment_index].values.astype(np.int64))[0]

                        current_boxes_overlay = viz.draw_bounding_boxes(person_id, frame, current_overall_bounding_box, current_person_color, boxes_thickness)
                        frame = cv2.addWeighted(current_boxes_overlay, current_confidence, frame, 1 - current_confidence, 0)
                        if person_track.full_body_data.joints is not None:
                            frame = viz.draw_skeleton_joints(frame, current_person_color, current_joints_radius, current_keypoints)
                            frame = viz.draw_skeleton_bones(frame, current_person_color, current_bones_thickness, current_keypoints)
                else:
                    if person_track.tracked_data.frames_indices.is_in_vicinity(video_reader.current_frame_index, self.frames_stride, mode='right'):
                        current_bounding_box = person_track.tracked_data.bounding_box(video_reader.current_frame_index)
                        current_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_bounding_box.values.astype(np.int64))[0]
                        current_keypoints = person_track.tracked_data.frame_keypoints(video_reader.current_frame_index)
                        current_confidence = person_track.tracked_data.confidence(video_reader.current_frame_index)

                        current_boxes_overlay = viz.draw_bounding_boxes(person_id, frame, current_bounding_box, current_person_color, boxes_thickness)
                        frame = cv2.addWeighted(current_boxes_overlay, current_confidence, frame, 1 - current_confidence, 0)
                        if person_track.tracked_data.joints is not None:
                            frame = viz.draw_skeleton_joints(frame, current_person_color, current_joints_radius, current_keypoints)
                            frame = viz.draw_skeleton_bones(frame, current_person_color, current_bones_thickness, current_keypoints)



            frame = cv2.resize(frame, video_width_height)
            if write_video: video_writer.write(frame)
            if show_frames:
                cv2.imshow(video_filename, frame)
                cv2.waitKey(show_frame_delay)

        if write_video: video_writer.release()
        if show_frames: cv2.destroyAllWindows()


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

    def visualize_partial_body_tracks(self, **parameters) -> None:
        pass
