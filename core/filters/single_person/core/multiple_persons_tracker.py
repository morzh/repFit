import os
import cv2
from ultralytics import YOLO

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.video_stride_reader import VideoStrideReader


class PersonsTracker:
    """
    Description:
        Tracker of person's data (idm bounding boxes, confidences) within given video.

    :ivar model: AI model for person(s) detection.
    """
    def __init__(self, weights_pathname: str = 'yolov10x.pt'):
        self.model = YOLO(weights_pathname)

    def track(self, video_filepath: os.PathLike, stride=2) -> MultiplePersonsTracks:
        """
        Description:
            Track persons, their ids, bounding boxes and their confidences in given video.
            Data will be stored in ``MultiplePersonsTracks`` class instance.

        :param video_filepath:  filepath of the input video;
        :param stride: video frames stride.

        :return: multiple persons tracks data.
        """
        if not os.path.isfile(video_filepath):
            raise Exception(f"Video {video_filepath} was not found")

        video_reader = VideoStrideReader(video_filepath, stride=stride, use_tqdm=False)
        persons_video_segments = MultiplePersonsTracks(video_properties=video_reader.video_properties)

        for frame in video_reader:
            predictions = self.model.track(frame, classes=0, persist=True, save=True, show=True, verbose=False)
            detected_data = predictions[0].boxes.data
            persons_video_segments.update(detected_data, video_reader.current_stride_frame_index)
            current_labeled_image_filepath = os.path.join(predictions[0].save_dir, predictions[0].path)
            current_labeled_image = cv2.imread(current_labeled_image_filepath)
            # video_writer.write(current_labeled_image)

        return persons_video_segments
