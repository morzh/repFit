import numpy as np
from pathlib import Path

from core.utils.cv.video_reader import VideoReader


class VideoReaderFramesBatch(VideoReader):
    """
    Use this class when you need stack of video frames at every loop iteration.
    """

    def __init__(self, video_filepath: str | Path, batch_size: int = 10, **options):
        """
        Description:
            VideoFramesBatch class constructor

        :param video_filepath: video filepath
        :param batch_size: number of video frames in stack to return by generator
        """
        super().__init__(video_filepath, **options)
        # self.video_filepath: str = video_filepath
        self.batch_size: int = batch_size
        # self.video_reader = VideoReader(video_filepath, use_tqdm=False)

    def __iter__(self):
        index = 0
        batch = None
        batch_size, height, width = self.batch_size, self.video_properties.height, self.video_properties.width
        while self.success:
            if not index:
                batch = np.empty((batch_size, height, width, 3))
            current_frame = self.read_frame()
            if self._original_frame_index % self.stride == 0:
                batch[index] = current_frame
                self._strided_frame_index += 1
                index += 1
                if index == self.batch_size:
                    index = 0
                    yield batch

            if index:
                yield batch[:index]
