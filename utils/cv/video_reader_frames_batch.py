import numpy as np
from pathlib import Path

from utils.cv.video_reader import VideoReader


class VideoReaderFramesBatch:
    """
    Use this class when you need stack of video frames at every loop iteration.
    """

    def __init__(self, video_filepath: str | Path, batch_size: int = 10):
        """
        Description:
            VideoFramesBatch class constructor

        :param video_filepath: video filepath
        :param batch_size: number of video frames in stack to return by generator
        """
        self.video_filepath: str = video_filepath
        self.batch_size: int = batch_size
        self.video_reader = VideoReader(video_filepath, use_tqdm=False)

    def __iter__(self):
        index = 0
        batch = None
        for frame in self.video_reader:
            if not index:
                batch = np.empty((self.batch_size, self.video_reader.height, self.video_reader.width, 3))
            batch[index] = frame
            index += 1
            if index == self.batch_size:
                index = 0
                yield batch
        if index:
            yield batch[:index]

    @property
    def fps(self):
        return self.video_reader.fps

    @property
    def resolution(self):
        return self.video_reader.resolution

    @property
    def width(self):
        return self.video_reader.width

    @property
    def height(self):
        return self.video_reader.height
