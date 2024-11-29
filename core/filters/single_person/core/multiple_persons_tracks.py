import cv2
import numpy as np
import os
import pickle

from cryptography.hazmat.primitives.asymmetric.ec import ECDSA

from core.filters.single_person.core.filters.multi_persons_filter_addon_base import MultiPersonsFilterAddonBase
from core.utils.cv.video_properties import VideoProperties
from core.filters.single_person.core.single_person_track import SinglePersonTrack
from core.utils.cv.video_reader import VideoReader
from core.utils.geometry.bounding_boxes_2d_array import BoundingBoxes2DArray


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


    def update(self, data: np.ndarray, frame_index: int) -> None:
        """
        Description:
            Update persons tracks data.

        :param data: person's data, obtained from AI model;
        :param frame_index: frame index.
        """

        for index in range(data.shape[0]):
            current_person_id = int(data[index, 4])

            if current_person_id == 0:
                """ No person ids tracked """
                continue
            elif current_person_id not in self.persons:
                self.persons[current_person_id] = SinglePersonTrack()

            current_confidence = float(data[index, 5])
            bounding_box = data[index, :4]
            self.persons[current_person_id].append(bounding_box, frame_index, confidence=current_confidence)


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
        :keyword next_frame_wait: gap between gaps in milliseconds
        :keyword hsv_step: HSV format hue circle step (in degrees)
        """
        boxes_thickness = options.get('frames_thickness', 2)
        next_frame_wait = options.get('next_frame_wait', 10)
        hsv_step = options.get('hsv_step', 20)
        confidence_transparent = options.get('transparent_threshold', 0.1)
        confidence_opaque = options.get('opaque_threshold', 0.95)

        confidence_range = confidence_opaque - confidence_transparent
        confidence_alpha_factor = 255.0 / confidence_range
        video_reader = VideoReader(self.video_properties.filepath)
        video_filename = os.path.basename(self.video_properties.filepath)
        imshow_window_name = video_filename  # .encode('ascii', 'ignore')

        for frame in video_reader:
            for person_id, person_track in self.persons.items():
                person_track.calculate_segments(self.frames_stride)
                for segment in person_track.segments:
                    if segment[0] <= video_reader.current_frame_index < segment[1]:
                        person_id_text = str(person_id).zfill(2)

                        current_bounding_box = self.persons[person_id].data.bounding_box(video_reader.current_frame_index)
                        current_bounding_box = BoundingBoxes2DArray.xywh_to_xyxy(current_bounding_box.astype(np.int64))[0]

                        current_confidence = self.persons[person_id].data.confidence(video_reader.current_frame_index)
                        current_alpha = self._visualize_alpha_from_confidence(confidence_transparent, confidence_opaque, current_confidence)
                        current_rgb_color = self._visualizing_color_cv2(hsv_step, person_id)

                        current_point_1 = tuple(current_bounding_box[:2])
                        current_point_2 = tuple(current_bounding_box[2:])
                        current_id_frame_point_2 = (current_point_1[0] + len(person_id_text) * 24, current_point_1[1] + 30)
                        current_text_point = (current_point_1[0] + 5, current_point_1[1] + 25)

                        current_overlay = frame.copy()
                        cv2.rectangle(current_overlay, current_point_1, current_point_2, current_rgb_color, boxes_thickness)
                        cv2.rectangle(current_overlay, current_point_1, current_id_frame_point_2, current_rgb_color, -1)
                        cv2.putText(current_overlay, person_id_text, current_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(200, 200, 200), thickness=6, lineType=cv2.LINE_AA)
                        cv2.putText(current_overlay, person_id_text, current_text_point, cv2.FONT_HERSHEY_DUPLEX, 1., color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                        frame = cv2.addWeighted(current_overlay, current_alpha, frame, 1 - current_alpha, 0)

            cv2.imshow(imshow_window_name, frame)
            cv2.waitKey(next_frame_wait)

        cv2.destroyAllWindows()


    @staticmethod
    def _visualizing_color_cv2(step, factor):
        hsv_color = np.uint8([[[(factor * step) % 180, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        rgb_color = rgb_color[0, 0]
        rgb_color = (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

        return  rgb_color

    @staticmethod
    def _visualize_alpha_from_confidence(minimum, maximum, confidence, normalized=True) -> int | float:
        values_range = maximum - minimum
        alpha_value = np.clip(confidence, minimum, maximum) - minimum
        alpha_value /= values_range
        if not normalized:
            alpha_value *= 255
            return int(alpha_value)
        return alpha_value

