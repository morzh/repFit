import numpy as np
from pathlib import Path
from typing import Iterator

from core.utils.cv.video_reader import VideoReader


class VideoReaderFramesBatch(VideoReader):
    """
    Use this class when you need stack of video frames at every loop iteration.
    """

    def __init__(self, video_filepath: str | Path, **options):
        """
        Description:
            VideoFramesBatch class constructor.

        :param video_filepath: video filepath

        :key batch_size: number of video frames in frames batches.
        """
        super().__init__(video_filepath, **options)
        self._batch_size: int = options.get('batch_size', 10)
        self._current_batch_index: int = -1


    def __iter__(self) -> Iterator:
        index = 0
        batch_size, height, width = self._batch_size, self.video_properties.height, self.video_properties.width
        batch = np.empty((batch_size, height, width, 3), dtype=np.uint8)

        for frame in super().__iter__():
            batch[index] = frame
            index += 1
            if index == self._batch_size:
                index = 0
                self._current_batch_index += 1
                yield batch
    
        last_batch_frames_number = self._current_frame_index % self._batch_size
        if last_batch_frames_number:
            self._current_batch_index += 1
            yield batch[:last_batch_frames_number + 1]


    @property
    def current_batch_index(self) -> int:
        """
        Description:
            Current index of video frames batch getter.
        """
        return self._current_batch_index


    @property
    def batch_size(self) -> int:
        """
        Description:
            Batch size getter.
        """
        return self._batch_size
