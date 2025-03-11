import os
from ultralytics import YOLO

from core.filters.single_person.source.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.video_stride_reader import VideoStrideReader


class PersonsTracker:
    """
    Description:
        Tracker of person's data (idm bounding boxes, confidences) within given video.

    :ivar model: AI model for person(s) detection.
    """
    def __init__(self, weights_pathname: str = 'yolov10l.pt'):
        self.model = YOLO(weights_pathname)


    def track(self, video_filepath: os.PathLike, **options) -> MultiplePersonsTracks:
        """
        Description:
            Track persons, their ids, bounding boxes and their confidences in given video.
            Data will be stored in ``MultiplePersonsTracks`` class instance.

        :param video_filepath:  filepath of the input video;

        :keyword frames_stride: video frames stride.
        :keyword visualize_while_tracking: show frames with info while tracking.
        :keyword verbose_tracking: show tracking information while tracking.

        :return: multiple persons tracks data.

        :raises IOError: if ``video_filepath`` does not exist.
        """
        if not os.path.isfile(video_filepath):
            raise IOError(f"Video {video_filepath} was not found")

        stride = options.get('frames_stride', 2)
        show_tracked_data = options.get('visualize_while_tracking', False)
        verbose = options.get('verbose_tracking', False)
        persist = options.get('tracking_persist', True)

        video_stride_reader = VideoStrideReader(video_filepath, stride=stride)
        persons_tracks = MultiplePersonsTracks(video_properties=video_stride_reader.video_properties, stride=stride)

        for frame in video_stride_reader:
            current_predictions = self.model.track(frame, classes=0, persist=persist, save=False, show=show_tracked_data, verbose=verbose)
            current_detected_bounding_boxes = current_predictions[0].boxes.data.cpu().numpy()
            current_detected_key_points = current_predictions[0].keypoints.data.cpu().numpy()
            current_frame_index = video_stride_reader.current_stride_frame_index

            if current_detected_bounding_boxes.size > 0 and current_detected_bounding_boxes.shape[1] != 7:  continue
            persons_tracks.update(current_frame_index, current_detected_bounding_boxes, keypoints=current_detected_key_points)

        return persons_tracks
