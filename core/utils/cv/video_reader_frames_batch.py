import numpy as np
from pathlib import Path

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
        :keyword use_tqdm: use console progress indicator.
        """
        options['stride'] = 1  # In case of batch reader frame stride is always equal to one.
        super().__init__(video_filepath, **options)
        self._batch_size: int = options.get('batch_size', 10)
        self._batch_frames_index_: int = -1

    def __getattribute__(self, attribute):
        """
        Description:
            Prevent accessing  super().current_stride_frame_index  property from the base ``VideoReader`` class.
        """
        if attribute == 'current_stride_frame_index':
            raise AttributeError

        return object.__getattribute__(self, attribute)

    def __iter__(self):
        index = 0
        batch_size, height, width = self._batch_size, self.video_properties.height, self.video_properties.width
        batch = np.empty((batch_size, height, width, 3), dtype=np.uint8)

        for frame in super().__iter__():
            batch[index] = frame
            index += 1
            if index == self._batch_size:
                index = 0
                self._batch_frames_index_ += 1
                yield batch

        last_batch_frames_number = self._current_source_frame_index % self._batch_size
        yield batch[:last_batch_frames_number]

    @property
    def current_batch_frame_index(self) -> int:
        return self._batch_frames_index_

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_frame_index(self) -> int:
        return self._batch_frames_index_
