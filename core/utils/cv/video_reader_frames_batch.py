import numpy as np
from pathlib import Path
from typing import Iterator

from core.utils.cv.video_reader_base import VideoReaderBase


class VideoReaderFramesBatch(VideoReaderBase):
    """
    Use this class when you need stack of video frames at every loop iteration.
    Violates Liskov substitution principle (__iter()__ returns [N, w, h, 3] array instead of [w, h, 3]).
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
        self._current_batch_index: int = -1



    def __iter__(self) -> Iterator:
        """
         Description:
            Frames generator without tqdm progress.

        :rtype: video frame
        """
        index = 0
        batch_size, height, width = self._batch_size, self.video_properties.height, self.video_properties.width
        batch = np.empty((batch_size, height, width, 3), dtype=np.uint8)

        while self._success:
            batch[index] = self._current_frame
            index += 1
            self._current_frame = self._read_frame()
            if index == self._batch_size:
                index = 0
                self._current_batch_index += 1
                yield batch

        last_batch_frames_number = self._current_frame_index % self._batch_size
        if last_batch_frames_number:
            self._current_batch_index += 1
            yield batch[:last_batch_frames_number + 1]

    '''
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
    '''

    @property
    def current_batch_index(self) -> int:
        return self._current_batch_index

    @property
    def batch_size(self) -> int:
        return self._batch_size
