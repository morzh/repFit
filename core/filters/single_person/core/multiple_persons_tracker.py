import os
import cv2
from ultralytics import YOLO

from core.filters.single_person.core.multiple_persons_tracks import MultiplePersonsTracks
from core.utils.cv.video_reader import VideoReader


class PersonsTracker:
    def __init__(self, weights_pathname: str = 'yolov10x.pt'):
        self.model_name = weights_pathname
        self.model = None
        self.video_reader = None

    def track(self, source_video_filepath: str, stride=2, target_video_folder: os.PathLike | str | None = None) -> MultiplePersonsTracks:
        if not os.path.isfile(source_video_filepath):
            raise Exception(f"Video {source_video_filepath} was not found")
        if target_video_folder is not None and not os.path.exists(target_video_folder):
            os.makedirs(target_video_folder, exist_ok=True)

        self.model = YOLO(self.model_name)
        self.video_reader = VideoReader(source_video_filepath, stride=2, use_tqdm=False)
        persons_video_segments = MultiplePersonsTracks()

        video_filename = os.path.basename(source_video_filepath)
        target_video_filepath = os.path.join(target_video_folder, video_filename)
        video_writer = cv2.VideoWriter(target_video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.video_reader.video_properties.fps, self.video_reader.video_properties.resolution)

        for frame in self.video_reader:
            predictions = self.model.track(frame, classes=0, persist=True, save=True, show=True, verbose=False)
            detected_data = predictions[0].boxes.data
            persons_video_segments.update(detected_data, self.video_reader.current_frame_index)
            current_labeled_image_filepath = os.path.join(predictions[0].save_dir, predictions[0].path)
            current_labeled_image = cv2.imread(current_labeled_image_filepath)
            # video_writer.write(current_labeled_image)

        video_writer.release()
        return persons_video_segments