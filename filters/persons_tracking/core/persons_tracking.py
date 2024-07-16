import os
import cv2
import numpy as np
from ultralytics import YOLO

from filters.persons_tracking.core.persons_video_segments import PersonsVideoSegments
from utils.cv.video_reader import VideoReader
from utils.cv.video_frames_batch import VideoFramesBatch


class PersonsTracker:
    def __init__(self, model_name: str = 'yolov10x.pt'):
        self.model = YOLO(model_name)
        self.detector_params = dict(
            classes=0,
            persist=True,
            conf=0.7,
            iou=0.7,
            show=False,
            tracker='tracker_conf.yaml',
            verbose=False
        )
        self.video_reader = None

    def extract_tracks(self, source_video_filepath: str, target_video_folder: str) -> PersonsVideoSegments:
        if not os.path.isfile(source_video_filepath):
            raise Exception(f"Video {source_video_filepath} was not found")
        if not os.path.exists(target_video_folder):
            os.makedirs(target_video_folder, exist_ok=True)

        persons_video_segments = PersonsVideoSegments()
        self.video_reader = VideoReader(source_video_filepath, use_tqdm=False)
        # self.video_reader = VideoFramesBatch(source_video_filepath)

        video_filename = os.path.basename(source_video_filepath)
        target_video_filepath = os.path.join(target_video_folder, video_filename)
        video_writer = cv2.VideoWriter(target_video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.video_reader.fps, self.video_reader.resolution)

        for frame in self.video_reader:
            results = self.model.track(frame, classes=0, persist=True, save=True, show=False)
            current_labeled_image_filepath = os.path.join(results[0].save_dir, results[0].path)
            current_labeled_image = cv2.imread(current_labeled_image_filepath)
            video_writer.write(current_labeled_image)

        video_writer.release()
        return persons_video_segments
