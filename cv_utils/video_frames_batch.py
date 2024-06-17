import numpy as np
from pathlib import Path

from cv_utils.video_reader import VideoReader


class VideoFramesBatch:
    def __init__(self, video_filepath: str | Path, batch_size: int = 10):
        self.video_filepath: str = video_filepath
        self.batch_size: int = batch_size
        self.video_reader = VideoReader(video_filepath, use_tqdm=False)

    def __iter__(self):
        batch = np.zeros((self.batch_size, self.video_reader.height, self.video_reader.width, 3))
        index = 0
        for frame in self.video_reader:
            batch[index] = frame
            index += 1
            if index == self.batch_size:
                index = 0
                yield batch
        video_frames_number = self.video_reader.current_frame_index
        last_batch_frames_number = self.video_reader.n_frames - int(video_frames_number / self.batch_size) * self.batch_size
        yield batch[:last_batch_frames_number]

