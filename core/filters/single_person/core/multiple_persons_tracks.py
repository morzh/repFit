import cv2
import numpy as np
import os
import pickle

from core.filters.single_person.core.filter_addons.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.core.single_person_track import SinglePersonTrack
from core.utils.cv.video_reader import VideoReader
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray
import core.utils.visualization.tracks_visualizing_utils as viz


class MultiplePersonsTracks:
    """
    Description:
        Persons data storage class.

    :ivar persons: person_id -> person data mapping
    :ivar video_properties:  video properties data
    """
    def __init__(self, video_properties: VideoProperties, stride=1):
        self.persons: dict[int, SinglePersonTrack] = {}
        self.video_properties = video_properties
        self.frames_number: int  = -1
        self.frames_stride = stride


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
                """ No person ids tracked """
                continue
            elif current_person_id not in self.persons:
                self.persons[current_person_id] = SinglePersonTrack()

            current_confidence = float(bounding_boxes[index, 5])
            current_bounding_box = bounding_boxes[index, :4]

            if keypoints is not None:
                current_keypoints = keypoints[index]

            self.persons[current_person_id].append(current_bounding_box, frame_index, confidence=current_confidence, keypoints=current_keypoints)


    def apply_filter(self, filter_visitor: MultiPersonsFilterAddonBase) -> None:
        """
        Description:
            Apply ``filter_visitor`` filter in place.

        :param filter_visitor: filter class instance.
        """
        filter_visitor.process(self)


    def serialize(self, filepath: os.PathLike | str) -> None:
        """
        Description:
            Serialize class instance.

        :param filepath: filepath to write class instance to.
        """
        with open(filepath, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    def visualize(self, **options) -> None:
        """
        Description:
            Visualize multiple persons tracks.

        :keyword frames_thickness: person bounding box thickness
        :keyword show_frame_delay: gap between gaps in milliseconds
        :keyword hsv_step: HSV format hue circle step (in degrees)
        """
        boxes_thickness = options.get('frames_thickness', 2)
        show_frame_delay = options.get('show_frame_delay', 10)
        hsv_step = options.get('hsv_step', 20)
        # confidence_transparent = options.get('transparent_threshold', 0.1)
        # confidence_opaque = options.get('opaque_threshold', 0.95)

        video_reader = VideoReader(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)
        imshow_window_name = video_filename  # .encode('ascii', 'ignore')

        for frame in video_reader:
            for person_id, person_track in self.persons.items():
                person_track.calculate_segments(self.frames_stride)
                mean_boxes_area = self.persons[person_id].mean_height()
                joints_radius = max(int(round(mean_boxes_area * 0.01)), 1)
                bones_thickness = max(int(round(joints_radius / 2)), 1)
                for segment in person_track.segments:
                    if segment[0] <= video_reader.current_frame_index < segment[1]:

                        current_bounding_box = self.persons[person_id].data.bounding_box(video_reader.current_frame_index)
                        current_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_bounding_box.astype(np.int64))[0]

                        current_keypoints = self.persons[person_id].data.frame_keypoints(video_reader.current_frame_index)

                        current_confidence = self.persons[person_id].data.confidence(video_reader.current_frame_index)
                        current_color = viz.stepped_color(hsv_step, person_id)
                        current_boxes_overlay = viz.draw_bounding_boxes(person_id, frame, current_bounding_box, current_color, boxes_thickness)

                        frame = cv2.addWeighted(current_boxes_overlay, current_confidence, frame, 1 - current_confidence, 0)
                        if person_track.data.keypoints is not None:
                            frame = viz.draw_skeleton_joints(frame, current_color, joints_radius, current_keypoints)
                            frame = viz.draw_skeleton_bones(frame, current_color, bones_thickness, current_keypoints)

            cv2.imshow(imshow_window_name, frame)
            cv2.waitKey(show_frame_delay)

        cv2.destroyAllWindows()

