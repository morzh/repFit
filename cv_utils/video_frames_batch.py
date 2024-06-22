import numpy as np
from pathlib import Path

from cv_utils.video_reader import VideoReader


class VideoFramesBatch:
    """
    Use this class when you need stack of video frames at every loop iteration.
    """
    def __init__(self, video_filepath: str | Path, batch_size: int = 10):
        """
        :param video_filepath: video filepath
        :param batch_size: number of video frames in stack to return by generator
        """
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
        if index:
            yield batch[:index]
