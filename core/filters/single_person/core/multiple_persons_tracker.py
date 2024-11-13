import os
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

        video_reader = VideoStrideReader(video_filepath, stride=stride)
        persons_tracks = MultiplePersonsTracks(video_properties=video_reader.video_properties)

        for frame in video_reader:
            predictions = self.model.track(frame, classes=0, persist=True, save=True, show=True, verbose=False)
            detected_data = predictions[0].boxes.data
            persons_tracks.update(detected_data, video_reader.current_frame_index)

        return persons_tracks
